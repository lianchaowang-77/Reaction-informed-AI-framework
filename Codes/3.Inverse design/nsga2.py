"""
Inverse design of acetals by NSGA-II (multi-objective) on discrete substituent-token space,
reusing the project's molwt-semantic evaluator + 51D pipeline (DNN3 seed3 oracle).

Goal:
  Input a target logk2 (point) or a logk2 range, and return top-K SMILES whose
  predicted logk2 best matches the target, under hard feasibility constraints.

Hard constraints (oracle pre-filter):
  SA < 5 and -1 < logP < 5

No novelty vs known sets (71w+184) is enforced, but we de-duplicate within a run.

Objectives (minimize):
  f1 = err (distance to target)
  f2 = sa_score
  f3 = |logP - logp_target|   (default logp_target=2.0)

Soft diversity:
  Optional final selection step for top-K using Morgan fingerprint Tanimoto penalty.

Outputs:
  Per run:
    - evaluated.csv
    - best_curve.csv (best-so-far min err vs oracle eval idx)
    - pareto_front_final.csv
    - topK.csv
    - meta.json
  Aggregated (out_dir root):
    - summary_runs.csv
    - final_unique_ranked.csv
    - final_topK.csv
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
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from evaluator import AcetalEvaluator, Gene, _canon_sub, EvalResult


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

    seed_i = 0
    chosen.append(seed_i)
    chosen_fp.append(fps[seed_i])
    remaining_idx.remove(seed_i)

    while len(chosen) < int(k_out) and remaining_idx:
        best_i = None
        best_score = -1e100
        for i in remaining_idx:
            err = float(pool.iloc[i]["err"])
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
    out = out.drop(columns=["_fp_ok"], errors="ignore")
    return out.reset_index(drop=True)


def _crossover(a: Gene, b: Gene, rng: random.Random) -> Tuple[Gene, Gene]:
    aa = [a.r1, a.r2, a.r3, a.r4]
    bb = [b.r1, b.r2, b.r3, b.r4]
    for i in range(4):
        if rng.random() < 0.5:
            aa[i], bb[i] = bb[i], aa[i]
    return Gene(*aa), Gene(*bb)


def _mutate(g: Gene, pools: Dict[str, List[str]], mut_prob: float, rng: random.Random) -> Gene:
    r = [g.r1, g.r2, g.r3, g.r4]
    keys = ["R1", "R2", "R3", "R4"]
    for i, key in enumerate(keys):
        if rng.random() < float(mut_prob):
            r[i] = rng.choice(pools[key])
    return Gene(*r)


def _gene_key(g: Gene) -> Tuple[str, str, str, str]:
    return (str(g.r1), str(g.r2), str(g.r3), str(g.r4))


def _invalid_genes_from_batch(batch: List[Gene], res: list) -> List[Gene]:
    valid = {_gene_key(r.gene) for r in res}
    return [g for g in batch if _gene_key(g) not in valid]


@dataclass
class Ind:
    r: EvalResult
    err: float
    f1: float
    f2: float
    f3: float
    rank: int = 10**9
    crowd: float = 0.0


def _dominates(a: Ind, b: Ind) -> bool:
    fa = (a.f1, a.f2, a.f3)
    fb = (b.f1, b.f2, b.f3)
    return all(x <= y for x, y in zip(fa, fb)) and any(x < y for x, y in zip(fa, fb))


def _fast_nondominated_sort(pop: List[Ind]) -> List[List[int]]:
    n = len(pop)
    S: List[List[int]] = [[] for _ in range(n)]
    n_dom = [0] * n
    fronts: List[List[int]] = []
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(pop[p], pop[q]):
                S[p].append(q)
            elif _dominates(pop[q], pop[p]):
                n_dom[p] += 1
    f0 = [i for i in range(n) if n_dom[i] == 0]
    fronts.append(f0)
    i = 0
    while fronts[i]:
        nxt: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    if fronts and not fronts[-1]:
        fronts.pop()
    return fronts


def _crowding_distance(pop: List[Ind], front: List[int]) -> None:
    if not front:
        return
    for i in front:
        pop[i].crowd = 0.0
    keys = [("f1", lambda x: x.f1), ("f2", lambda x: x.f2), ("f3", lambda x: x.f3)]
    for _, key in keys:
        fs = sorted(front, key=lambda i: key(pop[i]))
        pop[fs[0]].crowd = float("inf")
        pop[fs[-1]].crowd = float("inf")
        vmin = key(pop[fs[0]])
        vmax = key(pop[fs[-1]])
        if vmax == vmin:
            continue
        for j in range(1, len(fs) - 1):
            prev_v = key(pop[fs[j - 1]])
            next_v = key(pop[fs[j + 1]])
            pop[fs[j]].crowd += float(next_v - prev_v) / float(vmax - vmin)


def _assign_rank_and_crowd(pop: List[Ind]) -> None:
    fronts = _fast_nondominated_sort(pop)
    for r, front in enumerate(fronts):
        for i in front:
            pop[i].rank = int(r)
        _crowding_distance(pop, front)


def _tournament_nsga2(pop: List[Ind], rng: random.Random) -> Ind:
    a, b = rng.sample(pop, 2) if len(pop) >= 2 else (pop[0], pop[0])
    if a.rank < b.rank:
        return a
    if b.rank < a.rank:
        return b
    if a.crowd > b.crowd:
        return a
    if b.crowd > a.crowd:
        return b
    return a if rng.random() < 0.5 else b


def _pareto_front(pop: List[Ind]) -> List[Ind]:
    if not pop:
        return []
    fronts = _fast_nondominated_sort(pop)
    if not fronts:
        return []
    return [pop[i] for i in fronts[0]]


def run_one(
    run_id: int,
    evaluator: AcetalEvaluator,
    pools: Dict[str, List[str]],
    eval_budget: int,
    pop_size: int,
    cx_prob: float,
    mut_prob: float,
    rng: random.Random,
    batch_size: int,
    target: float,
    logp_target: float,
    k_out: int,
    diversity_lambda: float,
    diversity_fp_radius: int,
    diversity_fp_nbits: int,
    diversity_top_pool: int,
    convergence_cfg=None,
    history_rows: Optional[List[dict]] = None,
    convergence_threshold: Optional[float] = None,
    convergence_mode: str = "mean",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    evaluator.known = set()  # type: ignore[assignment]
    seen_run: set[str] = set()
    invalid_genes: set[Tuple[str, str, str, str]] = set()
    invalid_candidates = 0

    def propose_random(n: int) -> List[Gene]:
        out = []
        attempts = 0
        max_attempts = max(int(n) * 20, int(n))
        while len(out) < int(n) and attempts < max_attempts:
            attempts += 1
            g = Gene(rng.choice(pools["R1"]), rng.choice(pools["R2"]), rng.choice(pools["R3"]), rng.choice(pools["R4"]))
            if _gene_key(g) in invalid_genes:
                continue
            out.append(g)
        return out

    def calc_err(pred: float) -> float:
        return _err_point(pred, target=float(target))

    evaluated: List[Ind] = []
    best_curve = []
    best_err = math.inf
    rows_for_convergence: List[dict] = []
    history_rows = list(history_rows or [])
    converged = False
    convergence_status = {}

    def check_convergence() -> bool:
        nonlocal converged, convergence_status
        if convergence_cfg is None or convergence_threshold is None:
            return False
        ok, status, _ = evaluate_convergence(history_rows + rows_for_convergence, convergence_cfg, float(convergence_threshold), mode=str(convergence_mode))
        status["oracle_eval_idx"] = int(len(evaluated))
        convergence_status = status
        converged = bool(ok)
        return converged

    # init
    pop: List[Ind] = []
    while len(pop) < int(pop_size) and len(evaluated) < int(eval_budget) and not converged:
        batch = propose_random(max(400, pop_size))
        res = evaluator.evaluate_batch(batch, require_novel=False, require_feasible=True, seen_run=seen_run, batch_size=int(batch_size))
        for g in _invalid_genes_from_batch(batch, res):
            invalid_genes.add(_gene_key(g))
            invalid_candidates += 1
        for r in res:
            pred = float(r.pred_LOGk2_mean_seeds)
            sa = float(r.sa_score)
            lp = float(r.logP)
            err = float(calc_err(pred))
            ind = Ind(r=r, err=err, f1=err, f2=sa, f3=abs(lp - float(logp_target)))
            evaluated.append(ind)
            pop.append(ind)
            rows_for_convergence.append(rows_from_eval_result(r, err=err, evaluator=evaluator))
            best_err = min(best_err, err)
            best_curve.append({"oracle_eval_idx": len(evaluated), "best_err": float(best_err)})
            if len(pop) >= int(pop_size) or len(evaluated) >= int(eval_budget):
                break
        check_convergence()

    if not pop:
        raise RuntimeError("Failed to create initial feasible population; check pools/constraints.")

    while len(evaluated) < int(eval_budget) and not converged:
        _assign_rank_and_crowd(pop)
        children: List[Gene] = []
        child_attempts = 0
        max_child_attempts = max(int(pop_size) * 20, int(pop_size))
        while len(children) < int(pop_size) and child_attempts < max_child_attempts:
            child_attempts += 2
            p1 = _tournament_nsga2(pop, rng).r.gene
            p2 = _tournament_nsga2(pop, rng).r.gene
            if rng.random() < float(cx_prob):
                c1, c2 = _crossover(p1, p2, rng)
            else:
                c1, c2 = p1, p2
            c1 = _mutate(c1, pools, mut_prob, rng)
            c2 = _mutate(c2, pools, mut_prob, rng)
            if _gene_key(c1) not in invalid_genes:
                children.append(c1)
            if len(children) < int(pop_size):
                if _gene_key(c2) not in invalid_genes:
                    children.append(c2)

        new_inds: List[Ind] = []
        cursor = 0
        while len(new_inds) < int(pop_size) and len(evaluated) < int(eval_budget) and cursor < len(children):
            batch = children[cursor : cursor + 512]
            cursor += 512
            res = evaluator.evaluate_batch(batch, require_novel=False, require_feasible=True, seen_run=seen_run, batch_size=int(batch_size))
            for g in _invalid_genes_from_batch(batch, res):
                invalid_genes.add(_gene_key(g))
                invalid_candidates += 1
            for r in res:
                pred = float(r.pred_LOGk2_mean_seeds)
                sa = float(r.sa_score)
                lp = float(r.logP)
                err = float(calc_err(pred))
                ind = Ind(r=r, err=err, f1=err, f2=sa, f3=abs(lp - float(logp_target)))
                evaluated.append(ind)
                new_inds.append(ind)
                rows_for_convergence.append(rows_from_eval_result(r, err=err, evaluator=evaluator))
                best_err = min(best_err, err)
                best_curve.append({"oracle_eval_idx": len(evaluated), "best_err": float(best_err)})
                if len(new_inds) >= int(pop_size) or len(evaluated) >= int(eval_budget):
                    break
            if check_convergence():
                break

        # backfill
        backfill_rounds = 0
        while len(new_inds) < int(pop_size) and len(evaluated) < int(eval_budget) and backfill_rounds < 50 and not converged:
            backfill_rounds += 1
            batch = propose_random(512)
            res = evaluator.evaluate_batch(batch, require_novel=False, require_feasible=True, seen_run=seen_run, batch_size=int(batch_size))
            for g in _invalid_genes_from_batch(batch, res):
                invalid_genes.add(_gene_key(g))
                invalid_candidates += 1
            for r in res:
                pred = float(r.pred_LOGk2_mean_seeds)
                sa = float(r.sa_score)
                lp = float(r.logP)
                err = float(calc_err(pred))
                ind = Ind(r=r, err=err, f1=err, f2=sa, f3=abs(lp - float(logp_target)))
                evaluated.append(ind)
                new_inds.append(ind)
                rows_for_convergence.append(rows_from_eval_result(r, err=err, evaluator=evaluator))
                best_err = min(best_err, err)
                best_curve.append({"oracle_eval_idx": len(evaluated), "best_err": float(best_err)})
                if len(new_inds) >= int(pop_size) or len(evaluated) >= int(eval_budget):
                    break
            check_convergence()

        union = pop + new_inds
        _assign_rank_and_crowd(union)
        union.sort(key=lambda x: (x.rank, -x.crowd))
        pop = union[: int(pop_size)]

    df_eval = pd.DataFrame(
        [
            {
                "gene_R1_smiles": x.r.gene.r1,
                "gene_R2_smiles": x.r.gene.r2,
                "gene_R3_smiles": x.r.gene.r3,
                "gene_R4_smiles": x.r.gene.r4,
                "R1_smiles": evaluator.pos_cache.get(x.r.smiles, {}).get("R1_smiles", ""),
                "R2_smiles": evaluator.pos_cache.get(x.r.smiles, {}).get("R2_smiles", ""),
                "R3_smiles": evaluator.pos_cache.get(x.r.smiles, {}).get("R3_smiles", ""),
                "R4_smiles": evaluator.pos_cache.get(x.r.smiles, {}).get("R4_smiles", ""),
                "smiles": x.r.smiles,
                "sa_score": float(x.r.sa_score),
                "logP": float(x.r.logP),
                "pred_LOGk2_mean_seeds": float(x.r.pred_LOGk2_mean_seeds),
                "pred_LOGk2_std_seeds": float(getattr(x.r, "pred_LOGk2_std_seeds", float("nan"))),
                "err": float(x.err),
                "obj_f1_err": float(x.f1),
                "obj_f2_sa": float(x.f2),
                "obj_f3_abs_logP_minus_target": float(x.f3),
            }
            for x in evaluated
        ]
    )
    df_curve = pd.DataFrame(best_curve)

    front = _pareto_front(evaluated)
    df_front = pd.DataFrame(
        [
            {
                "smiles": x.r.smiles,
                "pred_LOGk2_mean_seeds": float(x.r.pred_LOGk2_mean_seeds),
                "pred_LOGk2_std_seeds": float(getattr(x.r, "pred_LOGk2_std_seeds", float("nan"))),
                "sa_score": float(x.r.sa_score),
                "logP": float(x.r.logP),
                "err": float(x.err),
                "obj_f1_err": float(x.f1),
                "obj_f2_sa": float(x.f2),
                "obj_f3_abs_logP_minus_target": float(x.f3),
            }
            for x in sorted(front, key=lambda z: z.err)
        ]
    )

    # unique ranked by err then pred
    df_eval["_pred_neg"] = -pd.to_numeric(df_eval["pred_LOGk2_mean_seeds"], errors="coerce")
    df_eval["_err"] = pd.to_numeric(df_eval["err"], errors="coerce")
    df_eval = df_eval.dropna(subset=["smiles", "_err", "pred_LOGk2_mean_seeds"]).copy()
    df_eval = df_eval.sort_values(["_err", "_pred_neg"], ascending=[True, True])
    df_uniq = df_eval.drop_duplicates(subset=["smiles"], keep="first").copy()
    df_uniq["err"] = df_uniq["_err"]
    df_uniq = df_uniq.drop(columns=["_pred_neg", "_err"], errors="ignore")
    df_uniq = df_uniq.sort_values(["err", "pred_LOGk2_mean_seeds"], ascending=[True, False])

    df_topk = _diverse_select(
        df_ranked=df_uniq,
        k_out=int(k_out),
        diversity_lambda=float(diversity_lambda),
        fp_radius=int(diversity_fp_radius),
        fp_nbits=int(diversity_fp_nbits),
        top_pool=int(diversity_top_pool),
    )

    meta = {
        "run_id": int(run_id),
        "eval_budget": int(eval_budget),
        "logp_target": float(logp_target),
        "attempted_unique_smiles": int(len(seen_run)),
        "n_unique_smiles": int(df_uniq["smiles"].nunique()),
        "best_err": float(df_uniq["err"].min()) if not df_uniq.empty else float("nan"),
        "best_by_err": df_uniq.head(1).to_dict(orient="records")[0] if not df_uniq.empty else {},
        "pareto_front_size": int(len(df_front)),
        "oracle_evals": int(len(evaluated)),
        "invalid_candidates": int(invalid_candidates),
        "invalid_gene_blacklist_size": int(len(invalid_genes)),
        "converged": bool(converged),
        "convergence_status": convergence_status,
    }
    return df_eval, df_curve, df_front, df_topk, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack_dir", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)

    # auto budget rule (same as inverse BO/GA)
    ap.add_argument("--runs", default=5, type=int)
    ap.add_argument("--evals_per_run", default=1000, type=int)

    ap.add_argument("--seed", default=1, type=int)
    ap.add_argument("--device", default="cpu", type=str)
    ap.add_argument("--batch_size", default=4096, type=int)

    # NSGA2 params
    ap.add_argument("--pop_size", default=0, type=int, help="If 0, auto-set based on eval budget.")
    ap.add_argument("--cx_prob", default=0.60, type=float)
    ap.add_argument("--mut_prob", default=0.25, type=float)
    ap.add_argument("--logp_target", default=2.0, type=float)

    # inverse target
    ap.add_argument("--target", default="6.099", type=str)
    ap.add_argument("--k_out", default=0, type=int)
    ap.add_argument("--pool_union_csv", default="", type=str, help="Override pool_union csv path (default: ./pool_union_for_inverse.csv).")

    # soft diversity
    ap.add_argument("--diversity_lambda", default=0.0, type=float)
    ap.add_argument("--diversity_fp_radius", default=2, type=int)
    ap.add_argument("--diversity_fp_nbits", default=1048, type=int)
    ap.add_argument("--diversity_top_pool", default=5000, type=int)
    add_convergence_args(ap)

    args = ap.parse_args()

    t_str = str(args.target).strip()
    if not t_str:
        raise ValueError("Specify --target for point-target inverse design.")
    target = float(t_str)
    rlo = rhi = None

    K = int(args.k_out)
    if int(args.evals_per_run) <= 0:
        evals_per_run = 1000 if K <= 100 else 10 * K
    else:
        evals_per_run = int(args.evals_per_run)

    if int(args.k_out) <= 0:
        k_out = 10
    else:
        k_out = int(args.k_out)

    if int(args.pop_size) <= 0:
        pop_size = min(200, max(40, int(evals_per_run // 5)))
    else:
        pop_size = int(args.pop_size)
    if pop_size >= evals_per_run:
        pop_size = max(20, int(evals_per_run // 2))

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

        df_eval, df_curve, df_front, df_topk, meta = run_one(
            run_id=int(r),
            evaluator=evaluator,
            pools=pools,
            eval_budget=int(evals_per_run),
            pop_size=int(pop_size),
            cx_prob=float(args.cx_prob),
            mut_prob=float(args.mut_prob),
            rng=rng,
            batch_size=int(args.batch_size),
            target=target,
            logp_target=float(args.logp_target),
            k_out=int(k_out),
            diversity_lambda=float(args.diversity_lambda),
            diversity_fp_radius=int(args.diversity_fp_radius),
            diversity_fp_nbits=int(args.diversity_fp_nbits),
            diversity_top_pool=int(args.diversity_top_pool),
            convergence_cfg=convergence_cfg,
            history_rows=history_rows,
            convergence_threshold=threshold,
            convergence_mode=mode,
        )

        eval_path = run_dir / "evaluated.csv"
        df_eval = df_eval.loc[:, ~df_eval.columns.duplicated()]
        df_eval.to_csv(eval_path, index=False, encoding="utf-8-sig")
        df_curve.to_csv(run_dir / "best_curve.csv", index=False, encoding="utf-8-sig")
        df_front.to_csv(run_dir / "pareto_front_final.csv", index=False, encoding="utf-8-sig")
        df_topk = df_topk.loc[:, ~df_topk.columns.duplicated()]
        df_topk.to_csv(run_dir / "topK.csv", index=False, encoding="utf-8-sig")
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

        best = meta.get("best_by_err", {}) or {}
        summary.append(
            {
                "run": int(r),
                "best_err": float(meta.get("best_err", float("nan"))),
                "best_pred_LOGk2_mean_seeds": float(best.get("pred_LOGk2_mean_seeds", float("nan"))),
                "best_smiles": str(best.get("smiles", "")),
                "best_sa_score": float(best.get("sa_score", float("nan"))),
                "best_logP": float(best.get("logP", float("nan"))),
                "pareto_front_size": int(meta.get("pareto_front_size", 0)),
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

    # aggregate across runs
    try:
        parts = []
        for p in eval_paths:
            dfp = pd.read_csv(p, encoding="utf-8-sig")
            dfp["run"] = int(p.parent.name.split("_")[1])
            parts.append(dfp)
        if parts:
            all_eval = pd.concat(parts, ignore_index=True)
            uniq = unique_ranked(all_eval, target=float(target))
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
        pass

    meta_all = {
        "runs_requested": int(args.runs),
        "runs_executed": int(len(summary)),
        "evals_per_run": int(evals_per_run),
        "total_oracle_evals": int(sum(int(x.get("oracle_evals", 0)) for x in summary)),
        "elapsed_sec": float(time.time() - t0),
        "params": vars(args),
        "adaptive_convergence": bool(args.adaptive_convergence),
        "auto_resolved": {"k_out": int(K), "evals_per_run": int(evals_per_run), "pop_size": int(pop_size)},
    }
    (out_dir / "meta_all.json").write_text(json.dumps(meta_all, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()












