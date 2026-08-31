"""
Monte Carlo Tree Search (MCTS) for inverse design: match a target logk2 (point or range),
using the same 51D feature pipeline and hard-constraint filter as GA.

Key semantics (matches GA setup):
- Search space: base+new substituent pools (pack/data/substituents/base_new/pool_union.csv)
- Candidate representation: (gene_R1_smiles..gene_R4_smiles) picked from pools
- Whole-molecule SMILES is constructed with the same RDKit assembly core:
    "[1*]OC([3*])([4*])O[2*]"
- "Final definition" uses the evaluator's molwt-standardized position assignment
  derived from whole-molecule SMILES (stored in evaluator.pos_cache), and is
  what the DNN3 seed3 model sees when building 51D.
- Hard constraints before oracle call:
    SA < 5, -1 < logP < 5, novelty vs known (71w+184) and vs current run
- Oracle: DNN3 5-seed ensemble, output columns pred_LOGk2_mean_seeds + pred_LOGk2_std_seeds.

Outputs (per run):
  - evaluated.csv: all oracle-evaluated candidates (with genes, assigned positions, smiles, SA/logP, pred_LOGk2_mean_seeds, pred_LOGk2_std_seeds)
- best_curve.csv: best-so-far (min err) vs oracle eval index
  - meta.json: run config and summary stats
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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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


@dataclass
class Node:
    depth: int
    parent: Optional["Node"]
    action: Optional[str]  # chosen substituent smiles for this depth (None for root)
    children: Dict[str, "Node"] = field(default_factory=dict)
    untried: List[str] = field(default_factory=list)
    n: int = 0
    w: float = 0.0  # value sum

    def q(self) -> float:
        return self.w / self.n if self.n > 0 else 0.0


def _uct_select(node: Node, c: float, rng: random.Random) -> Node:
    assert node.children, "cannot select from leaf"
    logN = math.log(max(1, node.n))

    best = None
    best_score = -1e100
    items = list(node.children.items())
    rng.shuffle(items)  # stable tie-breaking across Python dict order
    for _, ch in items:
        if ch.n == 0:
            score = 1e9
        else:
            score = ch.q() + float(c) * math.sqrt(logN / ch.n)
        if score > best_score:
            best_score = score
            best = ch
    assert best is not None
    return best


def _backprop(path: List[Node], reward: float) -> None:
    for n in path:
        n.n += 1
        n.w += float(reward)


def _sample_rollout_gene(pools: Dict[str, List[str]], chosen: Dict[str, str], rng: random.Random) -> Gene:
    r1 = chosen.get("R1") or rng.choice(pools["R1"])
    r2 = chosen.get("R2") or rng.choice(pools["R2"])
    r3 = chosen.get("R3") or rng.choice(pools["R3"])
    r4 = chosen.get("R4") or rng.choice(pools["R4"])
    return Gene(r1=r1, r2=r2, r3=r3, r4=r4)


def _node_path_to_partial(node: Node) -> Dict[str, str]:
    # Depth mapping: 1->R1, 2->R2, 3->R3, 4->R4
    pos_order = ["R1", "R2", "R3", "R4"]
    chosen: Dict[str, str] = {}
    cur = node
    while cur and cur.parent is not None:
        pos = pos_order[cur.depth - 1]
        if cur.action is not None:
            chosen[pos] = cur.action
        cur = cur.parent
    return chosen


def run_one(
    evaluator: AcetalEvaluator,
    pools: Dict[str, List[str]],
    eval_budget: int,
    uct_c: float,
    rng: random.Random,
    batch_size: int = 2048,
    max_attempt_factor: int = 50,
    sim_batch: int = 128,
    target: float = 6.099,
    invalid_reward: float = -10.0,
    convergence_cfg=None,
    history_rows: Optional[List[dict]] = None,
    convergence_threshold: Optional[float] = None,
    convergence_mode: str = "mean",
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    # Root has untried actions = pool for R1
    root = Node(depth=0, parent=None, action=None, untried=list(pools["R1"]))

    seen_run: set[str] = set()
    rows = []
    curve = []

    oracle_evals = 0
    attempts = 0
    best_err = math.inf
    history_rows = list(history_rows or [])
    converged = False
    convergence_status = {}

    # We count only oracle calls (i.e., returned EvalResult) toward eval_budget,
    # same definition as GA.
    pending_paths: List[List[Node]] = []
    pending_genes: List[Gene] = []

    def _flush_pending() -> None:
        nonlocal oracle_evals, best_err, converged, convergence_status
        if not pending_genes:
            return

        # Evaluate (hard constraints + oracle) in a batch
        res = evaluator.evaluate_batch(
            pending_genes,
            require_novel=False,
            require_feasible=True,
            seen_run=seen_run,
            batch_size=int(batch_size),
        )

        # Map evaluated results back to pending indices by gene tuple (unique within pending batch is expected)
        idx_map: Dict[Tuple[str, str, str, str], int] = {}
        for i, g in enumerate(pending_genes):
            idx_map[(g.r1, g.r2, g.r3, g.r4)] = i

        # Rejected candidates (duplicate/infeasible/evaluator failure) must be penalized.
        # Otherwise reward=0 is better than most valid inverse-design rewards (-err),
        # causing MCTS to prefer invalid branches and waste attempts.
        rewards = [float(invalid_reward) for _ in pending_genes]
        payload: Dict[int, object] = {}
        for r in res:
            key = (r.gene.r1, r.gene.r2, r.gene.r3, r.gene.r4)
            i = idx_map.get(key, -1)
            if i >= 0:
                pred = float(r.pred_LOGk2_mean_seeds)
                err = _err_point(pred, float(target))
                rewards[i] = -float(err)
                payload[i] = r

        # Backprop and record oracle-evaluated rows
        for i, path in enumerate(pending_paths):
            rew = float(rewards[i])
            _backprop(path, reward=rew)
            # Only now we count this as an oracle eval (same as GA)
            r = payload.get(i)
            if r is None:
                continue
            oracle_evals += 1
            pred = float(r.pred_LOGk2_mean_seeds)
            err = _err_point(pred, float(target))
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
                }
            )
            curve.append({"oracle_eval_idx": int(oracle_evals), "best_err": float(best_err)})

        pending_paths.clear()
        pending_genes.clear()
        if convergence_cfg is not None and convergence_threshold is not None:
            ok, status, _ = evaluate_convergence(history_rows + rows, convergence_cfg, float(convergence_threshold), mode=str(convergence_mode))
            status["oracle_eval_idx"] = int(oracle_evals)
            convergence_status = status
            converged = bool(ok)

    while oracle_evals < int(eval_budget) and attempts < int(eval_budget) * int(max_attempt_factor) and not converged:
        attempts += 1

        node = root
        path = [node]
        # Selection
        while node.depth < 4 and (not node.untried) and node.children:
            node = _uct_select(node, c=float(uct_c), rng=rng)
            path.append(node)

        # Expansion
        if node.depth < 4 and node.untried:
            a = node.untried.pop(rng.randrange(len(node.untried)))
            child_depth = node.depth + 1
            child = Node(depth=child_depth, parent=node, action=a)
            if child_depth < 4:
                pos_order = ["R1", "R2", "R3", "R4"]
                child.untried = list(pools[pos_order[child_depth]])
            node.children[a] = child
            node = child
            path.append(node)

        chosen = _node_path_to_partial(node)
        g = _sample_rollout_gene(pools, chosen, rng=rng)

        pending_paths.append(path)
        pending_genes.append(g)

        if len(pending_genes) >= int(sim_batch):
            _flush_pending()
            if oracle_evals >= int(eval_budget) or converged:
                break

    _flush_pending()

    df_eval = pd.DataFrame(rows)
    df_curve = pd.DataFrame(curve)
    meta = {
        "eval_budget": int(eval_budget),
        "oracle_evals": int(oracle_evals),
        "attempts_total": int(attempts),
        "uct_c": float(uct_c),
        "invalid_reward": float(invalid_reward),
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
        default=str(Path("datasets") / "acetal" / "pack" / "MCTS_results" / "mcts_5000eval_10runs_seed3_molwtSemantic"),
        help="Output directory for runs.",
    )
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--evals_per_run", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--uct_c", type=float, default=2.0, help="UCT exploration constant.")
    ap.add_argument("--batch_size", type=int, default=1048, help="Evaluator internal batch size.")
    ap.add_argument("--sim_batch", type=int, default=128, help="How many simulations to evaluate per batch (speed).")
    ap.add_argument("--invalid_reward", type=float, default=-10.0, help="Reward assigned to duplicate/infeasible candidates in MCTS backprop.")
    ap.add_argument("--target", type=float, default=6.099, help="Point target for inverse design.")
    ap.add_argument("--pool_union_csv", default="", type=str, help="Override pool_union csv path (default: ./pool_union_for_inverse.csv).")
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
            uct_c=float(args.uct_c),
            rng=rng,
            batch_size=int(args.batch_size),
            sim_batch=int(args.sim_batch),
            invalid_reward=float(args.invalid_reward),
            target=tgt,
            convergence_cfg=convergence_cfg,
            history_rows=history_rows,
            convergence_threshold=threshold,
            convergence_mode=mode,
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








