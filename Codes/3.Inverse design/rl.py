"""
Lightweight Reinforcement Learning (policy-gradient / REINFORCE) search for high predicted logk2 acetals,
using the same evaluator and hard-constraint filter as GA.

This is intentionally simple and robust for small discrete action spaces:
- Policy factorizes across positions: pi(R1)*pi(R2)*pi(R3)*pi(R4)
- Each action is selecting one substituent SMILES from the base+new pool for that position.
- Candidates are assembled -> hard constraints (SA/logP + novelty) -> oracle pred_LOGk2_mean_seeds.
- Update logits with REINFORCE using a running-mean baseline and optional entropy regularization.

Important: "final semantics" still come from the evaluator's molwt-standardized position assignment
based on whole-molecule SMILES (DNN3 seed3 is trained on that 51D pipeline).

Outputs (per run):
  - evaluated.csv
  - best_curve.csv
  - meta.json
Plus overall:
  - summary_runs.csv
  - meta_all.json
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure local pack directory is on sys.path for imports.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from evaluator import AcetalEvaluator, Gene, _canon_sub

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.warning")

@dataclass
class ConvergenceConfig:
    target: float
    top_n: int = 5
    history_keep: int = 20
    strict_rel_err: float = 0.01
    relaxed_rel_err: float = 0.05
    strict_runs: int = 10
    max_runs: int = 15


def add_convergence_args(ap) -> None:
    ap.add_argument("--adaptive_convergence", action="store_true", help="Enable cumulative top-N relative-error convergence.")
    ap.add_argument("--strict_rel_err", default=0.01, type=float, help="Strict per-candidate relative-error threshold for every molecule in cumulative top-N.")
    ap.add_argument("--relaxed_rel_err", default=0.05, type=float, help="Relaxed relative-error threshold used after strict runs fail.")
    ap.add_argument("--strict_runs", default=10, type=int, help="Number of runs using the strict threshold before fallback.")
    ap.add_argument("--max_runs", default=15, type=int, help="Maximum runs when adaptive convergence is enabled.")
    ap.add_argument("--converge_top_n", default=5, type=int, help="Number of unique candidates used for convergence.")
    ap.add_argument("--history_keep", default=20, type=int, help="Number of historical unique best candidates kept in memory.")


def config_from_args(args, target: float) -> ConvergenceConfig:
    return ConvergenceConfig(float(target), int(args.converge_top_n), int(args.history_keep),
                             float(args.strict_rel_err), float(args.relaxed_rel_err),
                             int(args.strict_runs), int(args.max_runs))


def canonical_smiles(smiles: object) -> str:
    s = str(smiles).strip()
    if not s:
        return ""
    mol = Chem.MolFromSmiles(s)
    return s if mol is None else Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def rel_err_denominator(target: float) -> float:
    value = abs(float(target))
    return max(value, 1e-10) if value <= 1e-5 else value


def _as_records(rows: object) -> List[dict]:
    if rows is None:
        return []
    if isinstance(rows, pd.DataFrame):
        return rows.to_dict(orient="records")
    return [dict(row) for row in rows]


def unique_ranked(rows: object, target: float) -> pd.DataFrame:
    records = _as_records(rows)
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    required = {"smiles", "err", "pred_LOGk2_mean_seeds"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    frame = frame.copy()
    frame["err"] = pd.to_numeric(frame["err"], errors="coerce")
    frame["pred_LOGk2_mean_seeds"] = pd.to_numeric(frame["pred_LOGk2_mean_seeds"], errors="coerce")
    frame = frame.dropna(subset=["smiles", "err", "pred_LOGk2_mean_seeds"]).copy()
    if frame.empty:
        return frame
    frame["canonical_smiles"] = frame["smiles"].map(canonical_smiles)
    frame = frame[frame["canonical_smiles"].astype(str).str.len() > 0].copy()
    if frame.empty:
        return frame
    frame["_pred_neg"] = -frame["pred_LOGk2_mean_seeds"]
    frame = frame.sort_values(["err", "_pred_neg"], ascending=[True, True])
    frame = frame.drop_duplicates(subset=["canonical_smiles"], keep="first").copy()
    frame["relative_err"] = frame["err"].astype(float) / rel_err_denominator(target)
    frame = frame.drop(columns=["_pred_neg"], errors="ignore")
    return frame.sort_values(["err", "pred_LOGk2_mean_seeds"], ascending=[True, False]).reset_index(drop=True)


def trim_history(rows: object, cfg: ConvergenceConfig) -> List[dict]:
    ranked = unique_ranked(rows, cfg.target)
    return [] if ranked.empty else ranked.head(cfg.history_keep).to_dict(orient="records")


def evaluate_convergence(rows: object, cfg: ConvergenceConfig, threshold: float,
                         mode: str = "mean") -> Tuple[bool, dict, pd.DataFrame]:
    mode = str(mode).strip().lower()
    if mode not in {"all", "mean"}:
        raise ValueError(f"bad convergence mode: {mode}")
    ranked = unique_ranked(rows, cfg.target)
    top = ranked.head(cfg.top_n).copy() if not ranked.empty else pd.DataFrame()
    if len(top) < cfg.top_n:
        return False, {"converged": False, "threshold": float(threshold), "mode": mode,
                       "top_n": cfg.top_n, "top_n_available": len(top),
                       "top_n_mean_relative_err": math.nan, "top_n_max_relative_err": math.nan}, top
    mean_rel = float(top["relative_err"].mean())
    max_rel = float(top["relative_err"].max())
    converged = bool(max_rel <= threshold) if mode == "all" else bool(mean_rel <= threshold)
    status = {"converged": converged, "threshold": float(threshold), "mode": mode,
              "top_n": cfg.top_n, "top_n_available": len(top),
              "top_n_mean_relative_err": mean_rel, "top_n_max_relative_err": max_rel,
              "best_err": float(top["err"].min()),
              "best_relative_err": float(top["relative_err"].min())}
    return converged, status, top


def active_threshold(run_id: int, cfg: ConvergenceConfig) -> float:
    return cfg.strict_rel_err if run_id <= cfg.strict_runs else cfg.relaxed_rel_err


def active_mode(run_id: int, cfg: ConvergenceConfig) -> str:
    return "all" if run_id <= cfg.strict_runs else "mean"


def pre_relaxed_decision(rows: object, cfg: ConvergenceConfig) -> Tuple[bool, dict, pd.DataFrame]:
    ok_all, status_all, top = evaluate_convergence(rows, cfg, cfg.relaxed_rel_err, "all")
    status_all["phase"] = "pre_relaxed_all"
    if ok_all:
        return True, status_all, top
    ok_mean, status_mean, top = evaluate_convergence(rows, cfg, cfg.relaxed_rel_err, "mean")
    status_mean["phase"] = "pre_relaxed_mean"
    if ok_mean:
        return True, status_mean, top
    status_mean.update({"pre_relaxed_all_converged": False,
                        "pre_relaxed_mean_converged": False,
                        "phase": "pre_relaxed_failed"})
    return False, status_mean, top


def rows_from_eval_result(result, err: float, evaluator, phase: str = "") -> dict:
    pos = evaluator.pos_cache.get(result.smiles, {})
    row = {"gene_R1_smiles": result.gene.r1, "gene_R2_smiles": result.gene.r2,
           "gene_R3_smiles": result.gene.r3, "gene_R4_smiles": result.gene.r4,
           "R1_smiles": pos.get("R1_smiles", ""), "R2_smiles": pos.get("R2_smiles", ""),
           "R3_smiles": pos.get("R3_smiles", ""), "R4_smiles": pos.get("R4_smiles", ""),
           "smiles": result.smiles, "canonical_smiles": canonical_smiles(result.smiles),
           "sa_score": float(result.sa_score), "logP": float(result.logP),
           "pred_LOGk2_mean_seeds": float(result.pred_LOGk2_mean_seeds),
           "pred_LOGk2_std_seeds": float(getattr(result, "pred_LOGk2_std_seeds", float("nan"))),
           "err": float(err)}
    if phase:
        row["phase"] = str(phase)
    return row

R_COLS = ["R1_smiles", "R2_smiles", "R3_smiles", "R4_smiles"]


def _err_point(pred: float, target: float) -> float:
    return abs(float(pred) - float(target))


def _err_range(pred: float, lo: float, hi: float) -> float:
    p = float(pred)
    lo = float(lo)
    hi = float(hi)
    if p < lo:
        return lo - p
    if p > hi:
        return p - hi
    return 0.0


def _default_pool_union_csv() -> Path:
    return Path(__file__).resolve().parent / "pool_union_for_inverse.csv"


def _target_anchor_value(target: Optional[float], range_lo: Optional[float], range_hi: Optional[float]) -> float:
    if target is not None:
        return float(target)
    if range_lo is not None and range_hi is not None:
        lo = float(range_lo)
        hi = float(range_hi)
        if hi < 2.0:
            return 1.5
        if lo >= 2.0 and hi <= 4.0:
            return 3.0
        if lo > 4.0:
            return 4.5
        return 0.5 * (lo + hi)
    return 4.5


def _pick_primary_source(anchor: float) -> str:
    if float(anchor) < 2.0:
        return "low2"
    if float(anchor) <= 4.0:
        return "middle2to4"
    return "high4"


def _load_pools(
    pack: Path,
    target_anchor: float,
    pool_union_csv: Optional[Path] = None,
) -> Dict[str, List[str]]:
    pool_csv = Path(pool_union_csv) if pool_union_csv is not None else _default_pool_union_csv()
    if not pool_csv.exists():
        raise FileNotFoundError(str(pool_csv))
    pdf = pd.read_csv(pool_csv, encoding="utf-8-sig")
    if "source" not in pdf.columns:
        raise ValueError(f"{pool_csv} missing required column: source")
    primary = _pick_primary_source(float(target_anchor))
    pools: Dict[str, List[str]] = {}
    for pos in ["R1", "R2", "R3", "R4"]:
        pri = pdf[(pdf["position"] == pos) & (pdf["source"] == primary)]["substituent"].astype(str).str.strip().tolist()
        exp = pdf[(pdf["position"] == pos) & (pdf["source"] == "experiment")]["substituent"].astype(str).str.strip().tolist()
        arr = pri + exp
        arr = [x for x in arr if x and x != "[H]"]
        seen = set()
        out = []
        for s in arr:
            c = _canon_sub(s)
            if not c or c in seen:
                continue
            seen.add(c)
            out.append(c)
        if not out:
            raise RuntimeError(f"empty pool for {pos} (primary={primary}, csv={pool_csv})")
        pools[pos] = out
    return pools


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    t = float(max(1e-6, temperature))
    x = logits / t
    x = x - float(np.max(x))
    e = np.exp(x)
    s = float(np.sum(e))
    if not math.isfinite(s) or s <= 0:
        # fallback to uniform
        return np.ones_like(logits) / float(len(logits))
    return e / s


def _sample_categorical(p: np.ndarray, rng: random.Random) -> int:
    r = rng.random()
    acc = 0.0
    for i, v in enumerate(p.tolist()):
        acc += float(v)
        if r <= acc:
            return int(i)
    return int(len(p) - 1)


def _gene_key(g: Gene) -> Tuple[str, str, str, str]:
    return (str(g.r1), str(g.r2), str(g.r3), str(g.r4))


def run_one(
    evaluator: AcetalEvaluator,
    pools: Dict[str, List[str]],
    eval_budget: int,
    rng: random.Random,
    lr: float,
    entropy_coef: float,
    baseline_beta: float,
    temperature: float,
    sample_batch: int,
    batch_size: int = 2048,
    target: float = 6.099,
    convergence_cfg=None,
    history_rows: Optional[List[dict]] = None,
    convergence_threshold: Optional[float] = None,
    convergence_mode: str = "mean",
    invalid_reward: float = -10.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    pos_order = ["R1", "R2", "R3", "R4"]
    actions = {p: pools[p] for p in pos_order}
    sizes = {p: len(actions[p]) for p in pos_order}

    logits = {p: np.zeros((sizes[p],), dtype=np.float64) for p in pos_order}

    seen_run: set[str] = set()
    rows = []
    curve = []

    oracle_evals = 0
    attempts_batches = 0
    best_err = math.inf

    baseline = 0.0
    n_updates = 0
    invalid_candidates = 0
    invalid_policy_updates = 0
    history_rows = list(history_rows or [])
    converged = False
    convergence_status = {}

    while oracle_evals < int(eval_budget) and not converged:
        attempts_batches += 1

        # Sample a batch of candidate genes from current policy
        genes: List[Gene] = []
        chosen_idx: List[Dict[str, int]] = []
        probs = {p: _softmax(logits[p], temperature=float(temperature)) for p in pos_order}

        for _ in range(int(sample_batch)):
            pick = {}
            for p in pos_order:
                pick[p] = _sample_categorical(probs[p], rng=rng)
            chosen_idx.append(pick)
            genes.append(
                Gene(
                    r1=actions["R1"][pick["R1"]],
                    r2=actions["R2"][pick["R2"]],
                    r3=actions["R3"][pick["R3"]],
                    r4=actions["R4"][pick["R4"]],
                )
            )

        # Evaluate: returns only those that pass hard constraints and get pred_LOGk2_mean_seeds (oracle calls)
        res = evaluator.evaluate_batch(
            genes,
            require_novel=False,
            require_feasible=True,
            seen_run=seen_run,
            batch_size=int(batch_size),
        )

        # Map evaluated results back to sampled indices (by smiles+gene match, stable enough for small pools).
        # We build a dict from (gene tuple) to list of indices in this batch.
        idx_map: Dict[Tuple[str, str, str, str], List[int]] = {}
        for i, g in enumerate(genes):
            k = _gene_key(g)
            idx_map.setdefault(k, []).append(i)

        def apply_policy_update(pick: Dict[str, int], adv: float) -> None:
            for p in pos_order:
                one = np.zeros_like(logits[p])
                one[pick[p]] = 1.0
                logits[p] += float(lr) * float(adv) * (one - probs[p])
                if float(entropy_coef) > 0:
                    uni = np.ones_like(probs[p]) / float(len(probs[p]))
                    logits[p] += float(entropy_coef) * (uni - probs[p])

        # Update per evaluated candidate
        for r in res:
            if oracle_evals >= int(eval_budget):
                break
            oracle_evals += 1

            pred = float(r.pred_LOGk2_mean_seeds)
            err = _err_point(pred, float(target))
            reward = -float(err)
            if n_updates == 0:
                baseline = reward
            else:
                baseline = float(baseline_beta) * float(baseline) + (1.0 - float(baseline_beta)) * reward
            adv = reward - float(baseline)

            # Find one matching sampled index for the gene (duplicates within batch can exist).
            key = _gene_key(r.gene)
            if key not in idx_map or not idx_map[key]:
                # Should be rare; skip update rather than corrupting indices.
                pass
            else:
                j = idx_map[key].pop()
                pick = chosen_idx[j]
                # REINFORCE update for each position logits: logits += lr * adv * (onehot - probs)
                apply_policy_update(pick, adv)
                n_updates += 1

            best_err = min(best_err, float(err))
            pos = evaluator.pos_cache.get(r.smiles, {})
            rows.append(
                {
                    "gene_R1_smiles": r.gene.r1,
                    "gene_R2_smiles": r.gene.r2,
                    "gene_R3_smiles": r.gene.r3,
                    "gene_R4_smiles": r.gene.r4,
                    "R1_smiles": pos.get("R1_smiles", ""),
                    "R2_smiles": pos.get("R2_smiles", ""),
                    "R3_smiles": pos.get("R3_smiles", ""),
                    "R4_smiles": pos.get("R4_smiles", ""),
                    "smiles": r.smiles,
                    "sa_score": float(r.sa_score),
                    "logP": float(r.logP),
                    "pred_LOGk2_mean_seeds": float(r.pred_LOGk2_mean_seeds),
                    "pred_LOGk2_std_seeds": float(r.pred_LOGk2_std_seeds),
                    "err": float(err),
                    "baseline": float(baseline),
                    "advantage": float(adv),
                }
            )
            curve.append({"oracle_eval_idx": int(oracle_evals), "best_err": float(best_err)})

        # Penalize sampled genes that were rejected by hard constraints or within-run de-duplication.
        # They do not count as oracle evaluations and are not written to evaluated.csv.
        invalid_indices = [i for indices in idx_map.values() for i in indices]
        invalid_candidates += len(invalid_indices)
        for j in invalid_indices:
            invalid_adv = float(invalid_reward) - float(baseline)
            apply_policy_update(chosen_idx[j], invalid_adv)
            invalid_policy_updates += 1

        if not res:
            # If nothing passes constraints, keep exploring by increasing temperature a bit.
            temperature = min(5.0, float(temperature) * 1.05)
            continue

        if convergence_cfg is not None and convergence_threshold is not None:
            ok, status, _ = evaluate_convergence(history_rows + rows, convergence_cfg, float(convergence_threshold), mode=str(convergence_mode))
            status["oracle_eval_idx"] = int(oracle_evals)
            convergence_status = status
            converged = bool(ok)

    df_eval = pd.DataFrame(rows)
    df_curve = pd.DataFrame(curve)
    meta = {
        "eval_budget": int(eval_budget),
        "oracle_evals": int(oracle_evals),
        "attempt_batches": int(attempts_batches),
        "lr": float(lr),
        "entropy_coef": float(entropy_coef),
        "baseline_beta": float(baseline_beta),
        "temperature_final": float(temperature),
        "sample_batch": int(sample_batch),
        "invalid_reward": float(invalid_reward),
        "invalid_candidates": int(invalid_candidates),
        "invalid_policy_updates": int(invalid_policy_updates),
        "best_err": float(best_err if best_err < 1e90 else float("nan")),
        "converged": bool(converged),
        "convergence_status": convergence_status,
    }
    return df_eval, df_curve, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pack_dir",
        type=str,
        default=str(Path("datasets") / "acetal" / "pack"),
        help="Pack directory containing models/schemas/props/known/pools.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(Path("datasets") / "acetal" / "pack" / "RL_results" / "rl_reinforce_5000eval_10runs_seed3_molwtSemantic"),
        help="Output directory for runs.",
    )
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--evals_per_run", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--entropy_coef", type=float, default=0.01)
    ap.add_argument("--baseline_beta", type=float, default=0.95, help="EMA factor for baseline (higher=slower).")
    ap.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature for action sampling.")
    ap.add_argument("--sample_batch", type=int, default=256, help="Number of candidates sampled per iteration (before filtering).")
    ap.add_argument("--batch_size", type=int, default=1048, help="Evaluator internal batch size.")
    ap.add_argument("--target", type=float, default=6.099, help="Point target for inverse design.")
    ap.add_argument("--pool_union_csv", default="", type=str, help="Override pool_union csv path (default: ./pool_union_for_inverse.csv).")
    ap.add_argument("--invalid_reward", type=float, default=-10.0, help="Negative reward used to penalize rejected sampled genes.")
    add_convergence_args(ap)

    args = ap.parse_args()

    pack = Path(args.pack_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tgt = float(args.target)
    rlo = rhi = None
    target_anchor = _target_anchor_value(target=tgt, range_lo=None, range_hi=None)
    pool_csv = Path(args.pool_union_csv).resolve() if str(args.pool_union_csv).strip() else None
    pools = _load_pools(pack, target_anchor=target_anchor, pool_union_csv=pool_csv)
    evaluator = AcetalEvaluator(pack_dir=pack, device=str(args.device))
    convergence_cfg = config_from_args(args, tgt) if bool(args.adaptive_convergence) else None
    run_limit = int(convergence_cfg.max_runs) if convergence_cfg is not None else int(args.runs)

    summary = []
    eval_paths = []
    t0 = time.time()
    history_rows: List[dict] = []
    convergence_events: List[dict] = []

    for r in range(1, int(run_limit) + 1):
        if convergence_cfg is not None and r == int(convergence_cfg.strict_runs) + 1:
            ok, status, top = pre_relaxed_decision(history_rows, convergence_cfg)
            status["run"] = int(r - 1)
            convergence_events.append(status)
            if ok:
                top.to_csv(out_dir / "convergence_top5.csv", index=False, encoding="utf-8-sig")
                break
        run_dir = out_dir / f"run_{r:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_t0 = time.time()
        rng = random.Random(int(args.seed) + r * 10007)
        threshold = active_threshold(r, convergence_cfg) if convergence_cfg is not None else None
        mode = active_mode(r, convergence_cfg) if convergence_cfg is not None else "mean"

        df_eval, df_curve, meta = run_one(
            evaluator=evaluator,
            pools=pools,
            eval_budget=int(args.evals_per_run),
            rng=rng,
            lr=float(args.lr),
            entropy_coef=float(args.entropy_coef),
            baseline_beta=float(args.baseline_beta),
            temperature=float(args.temperature),
            sample_batch=int(args.sample_batch),
            batch_size=int(args.batch_size),
            target=tgt,
            convergence_cfg=convergence_cfg,
            history_rows=history_rows,
            convergence_threshold=threshold,
            convergence_mode=mode,
            invalid_reward=float(args.invalid_reward),
        )

        df_eval = df_eval.loc[:, ~df_eval.columns.duplicated()]
        eval_path = run_dir / "evaluated.csv"
        df_eval.to_csv(eval_path, index=False, encoding="utf-8-sig")
        df_curve.to_csv(run_dir / "best_curve.csv", index=False, encoding="utf-8-sig")
        run_elapsed_sec = float(time.time() - run_t0)
        meta["runtime_sec"] = run_elapsed_sec
        (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        eval_paths.append(eval_path)
        if convergence_cfg is not None:
            history_rows = trim_history(history_rows + df_eval.to_dict(orient="records"), convergence_cfg)
            status = dict(meta.get("convergence_status", {}) or {})
            status["run"] = int(r)
            status["phase"] = "run_end"
            status["threshold_mode"] = "strict_all" if int(r) <= int(convergence_cfg.strict_runs) else "relaxed_mean"
            convergence_events.append(status)

        if not df_eval.empty:
            best_row = df_eval.sort_values("err", ascending=True).head(1).iloc[0].to_dict()
            summary.append(
                {
                    "run": int(r),
                    "best_err": float(best_row["err"]),
                    "best_pred_LOGk2_mean_seeds": float(best_row["pred_LOGk2_mean_seeds"]),
                    "best_pred_LOGk2_std_seeds": float(best_row["pred_LOGk2_std_seeds"]),
                    "best_smiles": str(best_row["smiles"]),
                    "best_sa_score": float(best_row["sa_score"]),
                    "best_logP": float(best_row["logP"]),
                    "oracle_evals": int(meta.get("oracle_evals", 0)),
                    "converged": bool(meta.get("converged", False)),
                    "top5_mean_relative_err": float((meta.get("convergence_status", {}) or {}).get("top_n_mean_relative_err", float("nan"))),
                    "runtime_sec": run_elapsed_sec,
                }
            )
        else:
            summary.append({"run": int(r), "best_err": float("nan"), "best_smiles": "", "best_sa_score": float("nan"), "best_logP": float("nan"), "oracle_evals": int(meta.get("oracle_evals", 0)), "runtime_sec": run_elapsed_sec})
        if convergence_cfg is not None and bool(meta.get("converged", False)):
            _, _, top = evaluate_convergence(history_rows, convergence_cfg, float(threshold), mode=str(mode))
            top.to_csv(out_dir / "convergence_top5.csv", index=False, encoding="utf-8-sig")
            break

    df_sum = pd.DataFrame(summary).sort_values("best_err", ascending=True)
    df_sum.to_csv(out_dir / "summary_runs.csv", index=False, encoding="utf-8-sig")
    if convergence_events:
        pd.DataFrame(convergence_events).to_csv(out_dir / "convergence_events.csv", index=False, encoding="utf-8-sig")
    # Aggregate across runs
    try:
        parts = []
        for p in eval_paths:
            if p.exists():
                parts.append(pd.read_csv(p))
        if parts:
            all_eval = pd.concat(parts, ignore_index=True)
            uniq = unique_ranked(all_eval.loc[:, ~all_eval.columns.duplicated()], target=float(tgt))
            uniq.to_csv(out_dir / "final_unique_ranked.csv", index=False, encoding="utf-8-sig")
            k_out = 10
            final_topk = uniq.head(int(k_out)).copy()
            final_topk.to_csv(out_dir / "final_topK.csv", index=False, encoding="utf-8-sig")
    except Exception:
        pass
    meta_all = {
        "runs_requested": int(args.runs),
        "runs_executed": int(len(summary)),
        "evals_per_run": int(args.evals_per_run),
        "total_oracle_evals": int(sum(int(x.get("oracle_evals", 0)) for x in summary)),
        "elapsed_sec": float(time.time() - t0),
        "params": vars(args),
        "adaptive_convergence": bool(args.adaptive_convergence),
    }
    (out_dir / "meta_all.json").write_text(json.dumps(meta_all, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()








