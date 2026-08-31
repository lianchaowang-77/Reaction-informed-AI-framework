"""
Inverse design of acetals by Bayesian Optimization (BO) on discrete substituent-token space,
reusing the project's molwt-semantic evaluator + 51D pipeline (DNN3 seed3 oracle).

Goal:
  Input a target logk2 (point) or a logk2 range, and return top-K SMILES whose
  predicted logk2 best matches the target, under hard feasibility constraints.

Important semantics:
- Search space (current stage): base+new substituent pools (pack/data/substituents/base_new/pool_union.csv)
  because cdft lookup is available for these substituents.
- Hard constraints enforced BEFORE oracle calls:
    SA < 5 and -1 < logP < 5
- Novelty vs known sets is NOT enforced by default (user request), but we still
  deduplicate within a run to avoid wasting budget.

BO details:
- Gene representation: (R1,R2,R3,R4) categorical choices
- Encoding: concatenated one-hot over tokens
- Surrogate: RandomForestRegressor predicting err (distance to target)
- Uncertainty: per-tree std (sigma)
- Acquisition (minimize err): maximize score = -(mu_err - kappa*sigma_err)
  i.e., prefer low predicted err while still exploring uncertain candidates.

Outputs (per run):
  - evaluated.csv: all oracle-evaluated candidates with pred/logP/SA + err
  - best_curve.csv: best-so-far (min err) vs oracle eval idx
  - topK.csv: final selected top-K candidates (optionally diversity-aware)
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
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor

from evaluator import AcetalEvaluator, Gene, _canon_sub


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


def _default_pool_union_csv() -> Path:
    # .../new_poll_inteligent_simplify/general_guided/inverse_design -> parents[2]=new_poll_inteligent_simplify
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
        arr = pri + exp  # priority: target-range source first, then experiment as supplement
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


def _parse_range(s: str) -> Tuple[float, float]:
    t = str(s).strip()
    if not t:
        raise ValueError("empty range")
    parts = [x.strip() for x in t.replace("[", "").replace("]", "").split(",")]
    if len(parts) != 2:
        raise ValueError(f"bad range '{s}', expected 'L,U'")
    lo = float(parts[0])
    hi = float(parts[1])
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def _err_point(pred: float, target: float) -> float:
    return abs(float(pred) - float(target))


def _err_range(pred: float, lo: float, hi: float) -> float:
    p = float(pred)
    if p < float(lo):
        return float(lo) - p
    if p > float(hi):
        return p - float(hi)
    return 0.0


def _gene_to_indices(g: Gene, idx: Dict[str, Dict[str, int]]) -> Tuple[int, int, int, int]:
    return (idx["R1"][g.r1], idx["R2"][g.r2], idx["R3"][g.r3], idx["R4"][g.r4])


def _gene_key(g: Gene) -> Tuple[str, str, str, str]:
    return (str(g.r1), str(g.r2), str(g.r3), str(g.r4))


def _invalid_genes_from_batch(batch: List[Gene], res: list) -> List[Gene]:
    valid = {_gene_key(r.gene) for r in res}
    return [g for g in batch if _gene_key(g) not in valid]


def _encode_onehot(idxs: List[Tuple[int, int, int, int]], sizes: Dict[str, int]) -> np.ndarray:
    n = len(idxs)
    d = sizes["R1"] + sizes["R2"] + sizes["R3"] + sizes["R4"]
    X = np.zeros((n, d), dtype=np.float32)
    off = {"R1": 0, "R2": sizes["R1"], "R3": sizes["R1"] + sizes["R2"], "R4": sizes["R1"] + sizes["R2"] + sizes["R3"]}
    for i, (a, b, c, d4) in enumerate(idxs):
        X[i, off["R1"] + int(a)] = 1.0
        X[i, off["R2"] + int(b)] = 1.0
        X[i, off["R3"] + int(c)] = 1.0
        X[i, off["R4"] + int(d4)] = 1.0
    return X


def _rf_predict_mu_sigma(model: RandomForestRegressor, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    preds = np.stack([t.predict(X) for t in model.estimators_], axis=0)
    mu = np.mean(preds, axis=0)
    sigma = np.std(preds, axis=0, ddof=0)
    return mu, sigma


def _propose_random(pools: Dict[str, List[str]], n: int, rng: random.Random) -> List[Gene]:
    out = []
    for _ in range(int(n)):
        out.append(Gene(rng.choice(pools["R1"]), rng.choice(pools["R2"]), rng.choice(pools["R3"]), rng.choice(pools["R4"])))
    return out


def _unique_genes(genes: List[Gene]) -> List[Gene]:
    seen = set()
    out = []
    for g in genes:
        k = (g.r1, g.r2, g.r3, g.r4)
        if k in seen:
            continue
        seen.add(k)
        out.append(g)
    return out


def _canon_fp(smi: str, radius: int, nbits: int) -> Optional[DataStructs.ExplicitBitVect]:
    m = Chem.MolFromSmiles(str(smi).strip())
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, int(radius), nBits=int(nbits))


def _diverse_select(
    df_ranked: pd.DataFrame,
    k_out: int,
    diversity_lambda: float,
    fp_radius: int,
    fp_nbits: int,
    top_pool: int,
) -> pd.DataFrame:
    """
    Soft diversity selection on top candidates.
    Greedy maximize: score = -err - lambda * max_sim(selected)
    """
    if int(k_out) <= 0:
        return df_ranked.head(0).copy()
    if float(diversity_lambda) <= 0:
        return df_ranked.head(int(k_out)).copy()

    pool = df_ranked.head(int(min(len(df_ranked), max(int(k_out), int(top_pool))))).copy()
    fps: List[Optional[DataStructs.ExplicitBitVect]] = []
    for smi in pool["smiles"].astype(str).tolist():
        fps.append(_canon_fp(smi, radius=int(fp_radius), nbits=int(fp_nbits)))
    pool["_fp_ok"] = [fp is not None for fp in fps]
    pool = pool[pool["_fp_ok"]].copy()
    fps = [fp for fp in fps if fp is not None]
    if pool.empty:
        return df_ranked.head(int(k_out)).copy()

    chosen = []
    chosen_fp: List[DataStructs.ExplicitBitVect] = []
    remaining_idx = list(range(len(pool)))

    # Seed with best err
    seed_i = 0
    chosen.append(seed_i)
    chosen_fp.append(fps[seed_i])
    remaining_idx.remove(seed_i)

    while len(chosen) < int(k_out) and remaining_idx:
        best_i = None
        best_score = -1e100
        for i in remaining_idx:
            err = float(pool.iloc[i]["err"])
            # max similarity to selected
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], chosen_fp)
            max_sim = float(max(sims)) if sims else 0.0
            score = -err - float(diversity_lambda) * max_sim
            if score > best_score:
                best_score = score
                best_i = i
        if best_i is None:
            break
        chosen.append(best_i)
        chosen_fp.append(fps[best_i])
        remaining_idx.remove(best_i)

    out = pool.iloc[chosen].copy()
    # restore original ranking order by err then pred preference (keep as chosen order is diversity-driven)
    out = out.drop(columns=["_fp_ok"], errors="ignore")
    return out.reset_index(drop=True)


def run_one(
    run_id: int,
    evaluator: AcetalEvaluator,
    pools: Dict[str, List[str]],
    eval_budget: int,
    n_init: int,
    n_candidates: int,
    propose_batch: int,
    kappa: float,
    rng: random.Random,
    batch_size: int,
    rf_trees: int,
    rf_max_depth: int,
    target: float,
    k_out: int,
    diversity_lambda: float,
    diversity_fp_radius: int,
    diversity_fp_nbits: int,
    diversity_top_pool: int,
    convergence_cfg=None,
    history_rows: Optional[List[dict]] = None,
    convergence_threshold: Optional[float] = None,
    convergence_mode: str = "mean",
    invalid_err_penalty: float = 10.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    # Important: do NOT enforce novelty vs known, but do de-duplicate within run.
    evaluator.known = set()  # type: ignore[assignment]
    seen_run: set[str] = set()

    idx: Dict[str, Dict[str, int]] = {p: {s: i for i, s in enumerate(pools[p])} for p in ["R1", "R2", "R3", "R4"]}
    sizes = {p: len(pools[p]) for p in ["R1", "R2", "R3", "R4"]}

    rows: List[dict] = []
    curve: List[dict] = []
    invalid_surrogate: Dict[Tuple[str, str, str, str], Gene] = {}
    invalid_candidates = 0

    best_err = math.inf
    converged = False
    convergence_status = {}
    history_rows = list(history_rows or [])

    def _calc_err(pred: float) -> float:
        return _err_point(pred, target=float(target))

    def _check_convergence() -> bool:
        nonlocal converged, convergence_status
        if convergence_cfg is None or convergence_threshold is None:
            return False
        ok, status, _ = evaluate_convergence(history_rows + rows, convergence_cfg, float(convergence_threshold), mode=str(convergence_mode))
        status["oracle_eval_idx"] = int(len(rows))
        convergence_status = status
        converged = bool(ok)
        return converged

    # Initial random feasible evaluations
    while len(rows) < int(n_init) and len(rows) < int(eval_budget):
        batch = _propose_random(pools, max(800, n_init), rng=rng)
        res = evaluator.evaluate_batch(batch, require_novel=False, require_feasible=True, seen_run=seen_run, batch_size=int(batch_size))
        for g in _invalid_genes_from_batch(batch, res):
            invalid_surrogate[_gene_key(g)] = g
            invalid_candidates += 1
        for r in res:
            pred = float(r.pred_LOGk2_mean_seeds)
            std = float(getattr(r, "pred_LOGk2_std_seeds", float("nan")))
            err = float(_calc_err(pred))
            pos = evaluator.pos_cache.get(r.smiles, {})
            row = {
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
                "pred_LOGk2_mean_seeds": pred,
                "pred_LOGk2_std_seeds": std,
                "err": err,
                "phase": "init_random",
            }
            rows.append(row)
            best_err = min(best_err, err)
            curve.append({"oracle_eval_idx": len(rows), "best_err": float(best_err)})
            if len(rows) >= int(n_init) or len(rows) >= int(eval_budget):
                break
        if _check_convergence():
            break

    if not rows:
        raise RuntimeError("Inverse BO init failed: no feasible candidates; check pools/constraints.")

    # BO loop
    it = 0
    while len(rows) < int(eval_budget) and not converged:
        it += 1

        # Fit surrogate: gene(onehot) -> err
        genes_eval = [Gene(r["gene_R1_smiles"], r["gene_R2_smiles"], r["gene_R3_smiles"], r["gene_R4_smiles"]) for r in rows]
        idxs_eval = [_gene_to_indices(g, idx) for g in genes_eval]
        X = _encode_onehot(idxs_eval, sizes=sizes)
        y = np.array([float(r["err"]) for r in rows], dtype=np.float32)
        if invalid_surrogate:
            invalid_genes = list(invalid_surrogate.values())
            invalid_idxs = [_gene_to_indices(g, idx) for g in invalid_genes]
            X_invalid = _encode_onehot(invalid_idxs, sizes=sizes)
            y_invalid = np.full((len(invalid_genes),), float(invalid_err_penalty), dtype=np.float32)
            X = np.vstack([X, X_invalid])
            y = np.concatenate([y, y_invalid])

        model = RandomForestRegressor(
            n_estimators=int(rf_trees),
            random_state=12345 + int(run_id) * 1000 + int(it),
            n_jobs=-1,
            max_depth=int(rf_max_depth) if int(rf_max_depth) > 0 else None,
            min_samples_leaf=1,
        )
        model.fit(X, y)

        # Candidate set for acquisition optimization
        cand_genes = _unique_genes(_propose_random(pools, int(n_candidates), rng=rng))
        cand_idxs = [_gene_to_indices(g, idx) for g in cand_genes]
        Xc = _encode_onehot(cand_idxs, sizes=sizes)
        mu, sigma = _rf_predict_mu_sigma(model, Xc)

        # minimize err via LCB; select smallest LCB -> maximize score = -LCB
        lcb = mu - float(kappa) * sigma
        score = -lcb
        order = np.argsort(-score)  # descending score

        remaining = int(eval_budget) - len(rows)
        target_new = min(int(propose_batch), remaining)

        proposed: List[Gene] = []
        for j in order[: max(3000, target_new * 10)]:
            proposed.append(cand_genes[int(j)])
            if len(proposed) >= max(3000, target_new * 10):
                break

        got_before = len(rows)
        cursor = 0
        backfill_rounds = 0
        while len(rows) - got_before < target_new and len(rows) < int(eval_budget):
            batch = proposed[cursor : cursor + 512]
            cursor += 512
            if not batch:
                backfill_rounds += 1
                if backfill_rounds > 30:
                    break
                batch = _propose_random(pools, 1024, rng=rng)

            res = evaluator.evaluate_batch(batch, require_novel=False, require_feasible=True, seen_run=seen_run, batch_size=int(batch_size))
            for g in _invalid_genes_from_batch(batch, res):
                invalid_surrogate[_gene_key(g)] = g
                invalid_candidates += 1
            for r in res:
                pred = float(r.pred_LOGk2_mean_seeds)
                std = float(getattr(r, "pred_LOGk2_std_seeds", float("nan")))
                err = float(_calc_err(pred))
                pos = evaluator.pos_cache.get(r.smiles, {})
                row = {
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
                    "pred_LOGk2_mean_seeds": pred,
                    "pred_LOGk2_std_seeds": std,
                    "err": err,
                    "phase": f"bo_iter_{it:04d}",
                }
                rows.append(row)
                best_err = min(best_err, err)
                curve.append({"oracle_eval_idx": len(rows), "best_err": float(best_err)})
                if len(rows) - got_before >= target_new or len(rows) >= int(eval_budget):
                    break
            if _check_convergence():
                break

    df_eval = pd.DataFrame(rows)
    df_curve = pd.DataFrame(curve)

    # Build unique feasible set and rank by err
    # Keep the row with minimal err per smiles; break ties by higher pred_LOGk2_mean_seeds.
    df_eval["_pred_neg"] = -pd.to_numeric(df_eval["pred_LOGk2_mean_seeds"], errors="coerce")
    df_eval["_err"] = pd.to_numeric(df_eval["err"], errors="coerce")
    df_eval = df_eval.dropna(subset=["smiles", "_err", "pred_LOGk2_mean_seeds"]).copy()
    df_eval = df_eval.sort_values(["_err", "_pred_neg"], ascending=[True, True])
    df_uniq = df_eval.drop_duplicates(subset=["smiles"], keep="first").copy()

    df_uniq["in_range"] = 0
    df_uniq["in_range"] = (df_uniq["_err"] <= 0.0).astype(int)
    df_uniq = df_uniq.sort_values(["_err", "pred_LOGk2_mean_seeds"], ascending=[True, False])

    df_uniq["err"] = df_uniq["_err"]
    df_ranked = df_uniq.drop(columns=["_pred_neg", "_err"], errors="ignore")

    df_topk = _diverse_select(
        df_ranked=df_ranked,
        k_out=int(k_out),
        diversity_lambda=float(diversity_lambda),
        fp_radius=int(diversity_fp_radius),
        fp_nbits=int(diversity_fp_nbits),
        top_pool=int(diversity_top_pool),
    )

    meta = {
        "run_id": int(run_id),
        "eval_budget": int(eval_budget),
        "n_init": int(n_init),
        "n_candidates": int(n_candidates),
        "propose_batch": int(propose_batch),
        "kappa": float(kappa),
        "rf_trees": int(rf_trees),
        "rf_max_depth": int(rf_max_depth),
        "target": float(target),
        "k_out": int(k_out),
        "diversity_lambda": float(diversity_lambda),
        "diversity_fp_radius": int(diversity_fp_radius),
        "diversity_fp_nbits": int(diversity_fp_nbits),
        "diversity_top_pool": int(diversity_top_pool),
        "attempted_unique_smiles": int(len(seen_run)),
        "n_unique_smiles": int(df_ranked["smiles"].nunique()),
        "best_err": float(df_ranked["err"].min()) if not df_ranked.empty else float("nan"),
        "best_by_err": df_ranked.head(1).to_dict(orient="records")[0] if not df_ranked.empty else {},
        "oracle_evals": int(len(rows)),
        "invalid_candidates": int(invalid_candidates),
        "invalid_surrogate_genes": int(len(invalid_surrogate)),
        "invalid_err_penalty": float(invalid_err_penalty),
        "converged": bool(converged),
        "convergence_status": convergence_status,
    }
    return df_eval, df_curve, df_topk, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack_dir", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    # Auto-budget rule (user requirement):
    # - If K <= 100: runs=5, evals_per_run=1000
    # - If K > 100: runs=5, evals_per_run=10*K
    # Users can override by explicitly passing --runs and/or --evals_per_run (>0).
    ap.add_argument("--runs", default=5, type=int, help="Number of independent runs (default 5).")
    ap.add_argument(
        "--evals_per_run",
        default=1000,
        type=int,
        help="Oracle eval budget per run. If 0, use auto rule based on K (see header comment).",
    )
    ap.add_argument("--seed", default=1, type=int)
    ap.add_argument("--device", default="cpu", type=str)
    ap.add_argument("--batch_size", default=4096, type=int)

    # BO params
    ap.add_argument(
        "--n_init",
        default=0,
        type=int,
        help="Initial random oracle evaluations. If 0, auto-set to ~0.2*evals_per_run (clipped).",
    )
    ap.add_argument("--n_candidates", default=10000, type=int)
    ap.add_argument(
        "--propose_batch",
        default=0,
        type=int,
        help="Target new oracle evaluations per BO iteration. If 0, auto-set to ~0.2*evals_per_run (clipped).",
    )
    ap.add_argument("--kappa", default=2.0, type=float)
    ap.add_argument("--rf_trees", default=300, type=int)
    ap.add_argument("--rf_max_depth", default=18, type=int)
    ap.add_argument("--invalid_err_penalty", default=10.0, type=float, help="Pseudo err assigned to rejected genes for BO surrogate training.")

    # inverse target
    ap.add_argument("--target", default="6.099", type=str, help="Point target logk2 (float).")
    ap.add_argument("--k_out", default=0, type=int, help="Number of candidates to output (0=auto: point->10, range->50).")
    ap.add_argument("--pool_union_csv", default="", type=str, help="Override pool_union csv path (default: ./pool_union_for_inverse.csv).")

    # soft diversity (final selection)
    ap.add_argument("--diversity_lambda", default=0.0, type=float, help="Soft penalty weight (0 disables).")
    ap.add_argument("--diversity_fp_radius", default=2, type=int)
    ap.add_argument("--diversity_fp_nbits", default=1048, type=int)
    ap.add_argument("--diversity_top_pool", default=5000, type=int, help="Apply diversity selection within top-P by err.")
    add_convergence_args(ap)

    args = ap.parse_args()

    t_str = str(args.target).strip()
    if not t_str:
        raise ValueError("Specify --target for point-target inverse design.")
    target = float(t_str)
    rlo = rhi = None

    # Auto budgeting based on k_out (K)
    K = int(args.k_out)
    if int(args.evals_per_run) <= 0:
        if K <= 100:
            evals_per_run = 1000
        else:
            evals_per_run = 10 * K
    else:
        evals_per_run = int(args.evals_per_run)

    if int(args.k_out) <= 0:
        k_out = 10
    else:
        k_out = int(args.k_out)

    # Auto n_init / propose_batch if unset
    def _clip(v: int, lo: int, hi: int) -> int:
        return max(int(lo), min(int(hi), int(v)))

    if int(args.n_init) <= 0:
        n_init = _clip(int(round(0.2 * evals_per_run)), lo=40, hi=500)
    else:
        n_init = int(args.n_init)

    if int(args.propose_batch) <= 0:
        propose_batch = _clip(int(round(0.2 * evals_per_run)), lo=40, hi=500)
    else:
        propose_batch = int(args.propose_batch)

    # Guard: init must be strictly less than budget so BO can iterate
    if n_init >= evals_per_run:
        n_init = max(1, int(evals_per_run // 2))
    if propose_batch <= 0:
        propose_batch = max(1, int(evals_per_run // 5))

    pack = Path(args.pack_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    target_anchor = _target_anchor_value(target=target, range_lo=None, range_hi=None)
    pool_csv = Path(args.pool_union_csv).resolve() if str(args.pool_union_csv).strip() else None
    pools = _load_pools(pack, target_anchor=target_anchor, pool_union_csv=pool_csv)
    evaluator = AcetalEvaluator(pack_dir=pack, device=str(args.device))
    convergence_cfg = config_from_args(args, target) if bool(args.adaptive_convergence) else None
    run_limit = int(convergence_cfg.max_runs) if convergence_cfg is not None else int(args.runs)

    summary = []
    t0 = time.time()
    topk_paths: List[Path] = []
    eval_paths: List[Path] = []
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

        df_eval, df_curve, df_topk, meta = run_one(
            run_id=int(r),
            evaluator=evaluator,
            pools=pools,
            eval_budget=int(evals_per_run),
            n_init=int(n_init),
            n_candidates=int(args.n_candidates),
            propose_batch=int(propose_batch),
            kappa=float(args.kappa),
            rng=rng,
            batch_size=int(args.batch_size),
            rf_trees=int(args.rf_trees),
            rf_max_depth=int(args.rf_max_depth),
            target=target,
            k_out=int(k_out),
            diversity_lambda=float(args.diversity_lambda),
            diversity_fp_radius=int(args.diversity_fp_radius),
            diversity_fp_nbits=int(args.diversity_fp_nbits),
            diversity_top_pool=int(args.diversity_top_pool),
            convergence_cfg=convergence_cfg,
            history_rows=history_rows,
            convergence_threshold=threshold,
            convergence_mode=mode,
            invalid_err_penalty=float(args.invalid_err_penalty),
        )

        df_eval = df_eval.loc[:, ~df_eval.columns.duplicated()]
        df_eval.to_csv(run_dir / "evaluated.csv", index=False, encoding="utf-8-sig")
        df_curve.to_csv(run_dir / "best_curve.csv", index=False, encoding="utf-8-sig")
        topk_path = run_dir / "topK.csv"
        eval_path = run_dir / "evaluated.csv"
        df_topk = df_topk.loc[:, ~df_topk.columns.duplicated()]
        df_topk.to_csv(topk_path, index=False, encoding="utf-8-sig")
        run_elapsed_sec = float(time.time() - run_t0)
        meta["runtime_sec"] = run_elapsed_sec
        (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        topk_paths.append(topk_path)
        eval_paths.append(eval_path)
        if convergence_cfg is not None:
            history_rows = trim_history(history_rows + df_eval.to_dict(orient="records"), convergence_cfg)
            status = dict(meta.get("convergence_status", {}) or {})
            status["run"] = int(r)
            status["phase"] = "run_end"
            status["threshold_mode"] = "strict_all" if int(r) <= int(convergence_cfg.strict_runs) else "relaxed_mean"
            convergence_events.append(status)

        best = meta.get("best_by_err", {}) or {}
        summary.append(
            {
                "run": int(r),
                "best_err": float(meta.get("best_err", float("nan"))),
                "best_pred_LOGk2_mean_seeds": float(best.get("pred_LOGk2_mean_seeds", float("nan"))),
                "best_smiles": str(best.get("smiles", "")),
                "best_sa_score": float(best.get("sa_score", float("nan"))),
                "best_logP": float(best.get("logP", float("nan"))),
                "oracle_evals": int(meta.get("oracle_evals", len(df_eval))),
                "converged": bool(meta.get("converged", False)),
                "top5_mean_relative_err": float((meta.get("convergence_status", {}) or {}).get("top_n_mean_relative_err", float("nan"))),
                "runtime_sec": run_elapsed_sec,
            }
        )
        if convergence_cfg is not None and bool(meta.get("converged", False)):
            _, _, top = evaluate_convergence(history_rows, convergence_cfg, float(threshold), mode=str(mode))
            top.to_csv(out_dir / "convergence_top5.csv", index=False, encoding="utf-8-sig")
            break

    pd.DataFrame(summary).sort_values("best_err", ascending=True).to_csv(out_dir / "summary_runs.csv", index=False, encoding="utf-8-sig")
    if convergence_events:
        pd.DataFrame(convergence_events).to_csv(out_dir / "convergence_events.csv", index=False, encoding="utf-8-sig")

    # Final aggregated output across runs:
    # - merge all evaluated rows
    # - dedup by smiles keeping minimal err (tie-break by higher pred)
    # - select top-K (optionally diversity-aware)
    try:
        parts = []
        for p in eval_paths:
            if p.exists():
                dfp = pd.read_csv(p, encoding="utf-8-sig")
                dfp["run"] = int(p.parent.name.split("_")[1])
                parts.append(dfp)
        if parts:
            all_eval = pd.concat(parts, ignore_index=True)
            uniq = unique_ranked(all_eval, target=float(target))
            uniq["in_range"] = 0
            uniq.to_csv(out_dir / "final_unique_ranked.csv", index=False, encoding="utf-8-sig")

            final_topk = _diverse_select(
                df_ranked=uniq,
                k_out=int(k_out),
                diversity_lambda=float(args.diversity_lambda),
                fp_radius=int(args.diversity_fp_radius),
                fp_nbits=int(args.diversity_fp_nbits),
                top_pool=int(args.diversity_top_pool),
            )
            final_topk = final_topk.loc[:, ~final_topk.columns.duplicated()]
            final_topk.to_csv(out_dir / "final_topK.csv", index=False, encoding="utf-8-sig")
    except Exception:
        # Do not fail the entire run if aggregation fails.
        pass

    meta_all = {
        "runs_requested": int(args.runs),
        "runs_executed": int(len(summary)),
        "evals_per_run": int(evals_per_run),
        "total_oracle_evals": int(sum(int(x.get("oracle_evals", 0)) for x in summary)),
        "elapsed_sec": float(time.time() - t0),
        "params": vars(args),
        "adaptive_convergence": bool(args.adaptive_convergence),
        "auto_resolved": {
            "k_out": int(K),
            "evals_per_run": int(evals_per_run),
            "n_init": int(n_init),
            "propose_batch": int(propose_batch),
        },
    }
    (out_dir / "meta_all.json").write_text(json.dumps(meta_all, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()











