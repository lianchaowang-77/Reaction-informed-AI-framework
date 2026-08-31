"""
NSGA-II (multi-objective evolutionary optimization) for high predicted logk2 acetals,
reusing the project's molwt-semantic evaluator + 51D pipeline.

Objectives (this variant):
  1) maximize pred_LOGk2_mean_seeds
  2) (disabled, constant)
  3) (disabled, constant)

Hard constraints and novelty are still enforced BEFORE oracle calls:
  SA < 5, -1 < logP < 5, novelty vs (71w+184) and vs current run.

Outputs (per run):
  - evaluated.csv: all oracle-evaluated candidates with objectives
  - best_curve.csv: best-so-far (by pred_LOGk2_mean_seeds) vs oracle eval idx
  - pareto_front_final.csv: non-dominated set from evaluated
  - meta.json
Plus overall:
  - summary_runs.csv
  - meta_all.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import RDLogger

from ga import AcetalEvaluator, Gene, _canon_sub
from guidance_simple import load_guidance, sample_guided_assignment_with_meta


@dataclass(frozen=True)
class GuidedProposal:
    gene: Any
    is_guided: bool = False
    guide_id: str = ""
    guide_weight: float = 0.0
    payload: Any = None


def _gene_key(gene: Any) -> Tuple[str, str, str, str]:
    return (str(gene.r1), str(gene.r2), str(gene.r3), str(gene.r4))


def quota_counts(total: int, guided_ratio: float) -> Tuple[int, int]:
    n = max(0, int(total))
    guided = int(round(n * min(1.0, max(0.0, float(guided_ratio)))))
    return guided, n - guided


def next_batch_target(actual_evals: int, eval_budget: int, batch_target: int) -> int:
    return min(max(0, int(batch_target)), max(0, int(eval_budget) - int(actual_evals)))


def evaluate_to_quota(*, evaluator: Any, target: int,
                      proposal_factory: Callable[[int], List[GuidedProposal]],
                      seen_run: set[str], batch_size: int,
                      max_rounds: int = 1000) -> List[Tuple[Any, GuidedProposal]]:
    wanted, accepted, rounds = max(0, int(target)), [], 0
    while len(accepted) < wanted:
        rounds += 1
        if rounds > int(max_rounds):
            raise RuntimeError(f"unable to fill effective evaluation quota: {len(accepted)}/{wanted}")
        unique, seen_keys = [], set()
        for proposal in proposal_factory(wanted - len(accepted)):
            key = _gene_key(proposal.gene)
            if key not in seen_keys:
                seen_keys.add(key)
                unique.append(proposal)
        if not unique:
            continue
        lookup = {_gene_key(p.gene): p for p in unique}
        results = evaluator.evaluate_batch([p.gene for p in unique], require_novel=True,
                                           require_feasible=True, seen_run=seen_run,
                                           batch_size=int(batch_size))
        for result in results:
            proposal = lookup.get(_gene_key(result.gene))
            if proposal is not None:
                accepted.append((result, proposal))
                if len(accepted) >= wanted:
                    break
    return accepted


def _row_value(row: Any, names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if isinstance(row, dict) and name in row:
            return row.get(name, default)
        if hasattr(row, name):
            return getattr(row, name)
    return default


@dataclass
class IterNoHitStopConfig:
    enabled: bool = False
    hit_threshold: float = 6.1
    patience_iters: int = 3
    min_evals: int = 0


def add_iter_nohit_stop_args(parser: Any) -> None:
    parser.add_argument("--iter_nohit_stop", action="store_true", help="Stop a run after consecutive algorithm iterations without high-value hits.")
    parser.add_argument("--iter_nohit_threshold", default=6.1, type=float)
    parser.add_argument("--iter_nohit_patience", default=3, type=int)
    parser.add_argument("--iter_nohit_min_evals", default=0, type=int)


def iter_nohit_config_from_args(args: Any) -> IterNoHitStopConfig:
    return IterNoHitStopConfig(bool(getattr(args, "iter_nohit_stop", False)),
                               float(getattr(args, "iter_nohit_threshold", 6.1)),
                               int(getattr(args, "iter_nohit_patience", 3)),
                               int(getattr(args, "iter_nohit_min_evals", 0)))


class IterNoHitStopper:
    def __init__(self, config: Optional[IterNoHitStopConfig]) -> None:
        self.config = config or IterNoHitStopConfig()
        self.iterations_checked = self.consecutive_nohit_iters = 0
        self.early_stopped = False
        self.stop_eval = self.stop_iteration = None
        self.stop_reason = ""

    def observe_iteration(self, new_rows: List[Any], n_evals: int, iteration: int) -> bool:
        cfg = self.config
        if not cfg.enabled or self.early_stopped or int(n_evals) <= int(cfg.min_evals):
            return False
        self.iterations_checked += 1
        hit = any((p := _row_value(r, ["pred_LOGk2_mean_seeds", "pred_mean"], None)) is not None
                  and float(p) > float(cfg.hit_threshold) for r in new_rows)
        self.consecutive_nohit_iters = 0 if hit else self.consecutive_nohit_iters + 1
        if self.consecutive_nohit_iters < int(cfg.patience_iters):
            return False
        self.early_stopped, self.stop_eval, self.stop_iteration = True, int(n_evals), int(iteration)
        self.stop_reason = f"no pred_LOGk2_mean_seeds > {cfg.hit_threshold} in {cfg.patience_iters} consecutive algorithm iterations"
        return True

    def to_meta(self, eval_budget: int, actual_evals: int) -> Dict[str, Any]:
        cfg = self.config
        return {"iter_nohit_stop_enabled": bool(cfg.enabled), "iter_nohit_threshold": float(cfg.hit_threshold),
                "iter_nohit_patience": int(cfg.patience_iters), "iter_nohit_min_evals": int(cfg.min_evals),
                "iter_nohit_early_stopped": bool(self.early_stopped), "iter_nohit_stop_eval": self.stop_eval,
                "iter_nohit_stop_iteration": self.stop_iteration, "iter_nohit_stop_reason": self.stop_reason,
                "iter_nohit_iterations_checked": int(self.iterations_checked),
                "iter_nohit_consecutive_nohit_iters_at_end": int(self.consecutive_nohit_iters),
                "actual_oracle_evals": int(actual_evals),
                "saved_oracle_evals": max(0, int(eval_budget) - int(actual_evals))}


RDLogger.DisableLog("rdApp.warning")


R_COLS = ["R1_smiles", "R2_smiles", "R3_smiles", "R4_smiles"]


def _load_pools(pack: Path) -> Dict[str, List[str]]:
    pool_csv = Path(__file__).resolve().parent / "pool_union_simplify_from_enrichment_practical.csv"
    if not pool_csv.exists():
        raise FileNotFoundError(str(pool_csv))
    pdf = pd.read_csv(pool_csv, encoding="utf-8-sig")
    pools: Dict[str, List[str]] = {}
    for pos in ["R1", "R2", "R3", "R4"]:
        arr = [str(x).strip() for x in pdf[pdf["position"] == pos]["substituent"].tolist()]
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
            raise RuntimeError(f"empty pool for {pos}")
        pools[pos] = out
    return pools


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


@dataclass
class Ind:
    gene: Gene
    smiles: str
    sa: float
    lp: float
    pred_mean: float
    pred_std: float
    # objective vector for minimization
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
    # per objective
    objs = [
        ("f1", lambda ind: ind.f1),
        ("f2", lambda ind: ind.f2),
        ("f3", lambda ind: ind.f3),
    ]
    for _, key in objs:
        front_sorted = sorted(front, key=lambda i: key(pop[i]))
        pop[front_sorted[0]].crowd = float("inf")
        pop[front_sorted[-1]].crowd = float("inf")
        vmin = key(pop[front_sorted[0]])
        vmax = key(pop[front_sorted[-1]])
        if vmax == vmin:
            continue
        for j in range(1, len(front_sorted) - 1):
            prev_v = key(pop[front_sorted[j - 1]])
            next_v = key(pop[front_sorted[j + 1]])
            pop[front_sorted[j]].crowd += float(next_v - prev_v) / float(vmax - vmin)


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
    # higher crowding wins
    if a.crowd > b.crowd:
        return a
    if b.crowd > a.crowd:
        return b
    return a if rng.random() < 0.5 else b


def _to_ind(evaluator: AcetalEvaluator, r, logp_target: float) -> Ind:
    # r is EvalResult from the shared GA/evaluator module.
    pred_mean = float(r.pred_LOGk2_mean_seeds)
    pred_std = float(r.pred_LOGk2_std_seeds)
    sa = float(r.sa_score)
    lp = float(r.logP)
    # single-objective variant: maximize pred_LOGk2_mean_seeds only
    f1 = -pred_mean
    f2 = 0.0
    f3 = 0.0
    return Ind(gene=r.gene, smiles=r.smiles, sa=sa, lp=lp, pred_mean=pred_mean, pred_std=pred_std, f1=f1, f2=f2, f3=f3)


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
    logp_target: float,
    guidance: Optional[Dict[str, List[dict]]] = None,
    guided_ratio: float = 0.30,
    iter_nohit_stop_config: Optional[IterNoHitStopConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object], pd.DataFrame]:
    seen_run: set[str] = set()
    evaluated: List[Ind] = []
    best_curve: List[dict] = []
    annotations: Dict[str, dict] = {}
    iter_nohit_stopper = IterNoHitStopper(iter_nohit_stop_config)
    stop_requested = False
    best = -math.inf

    def propose_random(n: int, *, force_guided: bool) -> List[GuidedProposal]:
        out: List[GuidedProposal] = []
        for _ in range(int(n)):
            if force_guided:
                a, meta = sample_guided_assignment_with_meta(pools, guidance, rng=rng)
                if not bool(meta.get("is_guided", False)):
                    raise RuntimeError("guided quota requested but no R3-R4 guidance was loaded")
                out.append(
                    GuidedProposal(
                        Gene(a["R1"], a["R2"], a["R3"], a["R4"]),
                        is_guided=True,
                        guide_id=str(meta.get("guide_id", "")),
                        guide_weight=float(meta.get("guide_weight", 0.0)),
                    )
                )
            else:
                out.append(
                    GuidedProposal(
                        Gene(
                            rng.choice(pools["R1"]),
                            rng.choice(pools["R2"]),
                            rng.choice(pools["R3"]),
                            rng.choice(pools["R4"]),
                        )
                    )
                )
        return out

    def propose_children(n: int, population: List[Ind]) -> List[GuidedProposal]:
        out: List[GuidedProposal] = []
        while len(out) < int(n):
            p1 = _tournament_nsga2(population, rng).gene
            p2 = _tournament_nsga2(population, rng).gene
            if rng.random() < float(cx_prob):
                c1, c2 = _crossover(p1, p2, rng)
            else:
                c1, c2 = p1, p2
            out.append(GuidedProposal(_mutate(c1, pools, mut_prob, rng)))
            if len(out) < int(n):
                out.append(GuidedProposal(_mutate(c2, pools, mut_prob, rng)))
        return out

    def record_pairs(
        pairs: List[Tuple[object, GuidedProposal]],
        *,
        phase: str,
    ) -> List[Ind]:
        nonlocal best, stop_requested
        recorded: List[Ind] = []
        for result, proposal in pairs:
            ind = _to_ind(evaluator, result, logp_target=logp_target)
            evaluated.append(ind)
            recorded.append(ind)
            annotations[ind.smiles] = {
                "phase": phase,
                "is_guided": bool(proposal.is_guided),
                "guide_id": str(proposal.guide_id),
                "guide_weight": float(proposal.guide_weight),
            }
            best = max(best, float(ind.pred_mean))
            best_curve.append({"oracle_eval_idx": len(evaluated), "best_pred_LOGk2_mean_seeds": best})
        return recorded

    # Initialize with an exact guided/unguided effective-evaluation quota.
    pop: List[Ind] = []
    init_target = next_batch_target(len(evaluated), int(eval_budget), int(pop_size))
    init_guided, init_unguided = quota_counts(init_target, guided_ratio)
    guided_pairs = evaluate_to_quota(
        evaluator=evaluator,
        target=init_guided,
        proposal_factory=lambda n: propose_random(n, force_guided=True),
        seen_run=seen_run,
        batch_size=int(batch_size),
    )
    pop.extend(record_pairs(guided_pairs, phase="init_guided"))
    unguided_pairs = evaluate_to_quota(
        evaluator=evaluator,
        target=init_unguided,
        proposal_factory=lambda n: propose_random(n, force_guided=False),
        seen_run=seen_run,
        batch_size=int(batch_size),
    )
    pop.extend(record_pairs(unguided_pairs, phase="init_unguided"))

    if not pop:
        raise RuntimeError("Failed to create initial feasible+novel population; check pools/constraints.")

    # NSGA-II loop
    gen = 0
    while len(evaluated) < int(eval_budget) and not stop_requested:
        gen += 1
        _assign_rank_and_crowd(pop)

        target_new = next_batch_target(len(evaluated), int(eval_budget), int(pop_size))
        target_guided, target_unguided = quota_counts(target_new, guided_ratio)
        new_inds: List[Ind] = []
        gen_rows: List[Ind] = []
        guided_pairs = evaluate_to_quota(
            evaluator=evaluator,
            target=target_guided,
            proposal_factory=lambda n: propose_random(n, force_guided=True),
            seen_run=seen_run,
            batch_size=int(batch_size),
        )
        guided_inds = record_pairs(guided_pairs, phase=f"nsga2_gen_{gen:03d}_guided")
        new_inds.extend(guided_inds)
        gen_rows.extend(guided_inds)
        unguided_pairs = evaluate_to_quota(
            evaluator=evaluator,
            target=target_unguided,
            proposal_factory=lambda n: propose_children(n, pop),
            seen_run=seen_run,
            batch_size=int(batch_size),
        )
        unguided_inds = record_pairs(unguided_pairs, phase=f"nsga2_gen_{gen:03d}_unguided")
        new_inds.extend(unguided_inds)
        gen_rows.extend(unguided_inds)

        # Environmental selection: parents + offspring
        union = pop + new_inds
        _assign_rank_and_crowd(union)
        # Sort by (rank asc, crowd desc)
        union.sort(key=lambda x: (x.rank, -x.crowd))
        pop = union[: int(pop_size)]
        if iter_nohit_stopper.observe_iteration(gen_rows, len(evaluated), gen):
            stop_requested = True

    # Build outputs
    df_eval = pd.DataFrame(
        [
            {
                "gene_R1_smiles": x.gene.r1,
                "gene_R2_smiles": x.gene.r2,
                "gene_R3_smiles": x.gene.r3,
                "gene_R4_smiles": x.gene.r4,
                "R1_smiles": evaluator.pos_cache.get(x.smiles, {}).get("R1_smiles", ""),
                "R2_smiles": evaluator.pos_cache.get(x.smiles, {}).get("R2_smiles", ""),
                "R3_smiles": evaluator.pos_cache.get(x.smiles, {}).get("R3_smiles", ""),
                "R4_smiles": evaluator.pos_cache.get(x.smiles, {}).get("R4_smiles", ""),
                "smiles": x.smiles,
                "sa_score": x.sa,
                "logP": x.lp,
                "pred_LOGk2_mean_seeds": x.pred_mean,
                "pred_LOGk2_std_seeds": x.pred_std,
                "obj_f1_neg_pred": x.f1,
                "obj_f2_sa": x.f2,
                "obj_f3_abs_logP_minus_target": x.f3,
                "phase": annotations.get(x.smiles, {}).get("phase", ""),
                "is_guided": bool(annotations.get(x.smiles, {}).get("is_guided", False)),
                "guide_id": annotations.get(x.smiles, {}).get("guide_id", ""),
                "guide_weight": float(annotations.get(x.smiles, {}).get("guide_weight", 0.0)),
            }
            for x in evaluated
        ]
    )
    df_curve = pd.DataFrame(best_curve)

    front = _pareto_front(evaluated)
    df_front = pd.DataFrame(
        [
            {
                "smiles": x.smiles,
                "pred_LOGk2_mean_seeds": x.pred_mean,
                "pred_LOGk2_std_seeds": x.pred_std,
                "sa_score": x.sa,
                "logP": x.lp,
                "obj_f1_neg_pred": x.f1,
                "obj_f2_sa": x.f2,
                "obj_f3_abs_logP_minus_target": x.f3,
            }
            for x in sorted(front, key=lambda z: z.pred_mean, reverse=True)
        ]
    )

    best_row = df_eval.sort_values("pred_LOGk2_mean_seeds", ascending=False).head(1).to_dict(orient="records")[0]
    meta = {
        "run_id": int(run_id),
        "eval_budget": int(eval_budget),
        "oracle_evals": int(len(evaluated)),
        "attempted_unique_smiles": int(len(seen_run)),
        "effective_evals_per_iteration": int(pop_size),
        "guided_ratio_requested": float(guided_ratio),
        "guided_oracle_evals": int(sum(bool(annotations.get(x.smiles, {}).get("is_guided", False)) for x in evaluated)),
        "logp_target": float(logp_target),
        "best": best_row,
        "pareto_front_size": int(len(df_front)),
    }
    meta["unguided_oracle_evals"] = int(len(evaluated) - int(meta["guided_oracle_evals"]))
    meta["effective_guided_oracle_ratio"] = float(meta["guided_oracle_evals"] / len(evaluated))
    meta.update(iter_nohit_stopper.to_meta(eval_budget=int(eval_budget), actual_evals=len(evaluated)))
    return df_eval, df_curve, meta, df_front


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack_dir", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--runs", default=10, type=int)
    ap.add_argument("--evals_per_run", default=5000, type=int)
    ap.add_argument("--pop_size", default=200, type=int)
    ap.add_argument("--cx_prob", default=0.60, type=float)
    ap.add_argument("--mut_prob", default=0.25, type=float)
    ap.add_argument("--logp_target", default=2.0, type=float, help="Third objective is abs(logP - target).")
    ap.add_argument("--seed", default=1, type=int)
    ap.add_argument("--device", default="cpu", type=str)
    ap.add_argument("--batch_size", default=4096, type=int)
    ap.add_argument("--combo_dir", default="", type=str)
    ap.add_argument("--guided_ratio", default=0.30, type=float)
    add_iter_nohit_stop_args(ap)
    args = ap.parse_args()

    pack = Path(args.pack_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pools = _load_pools(pack)
    guidance = None
    if str(args.combo_dir).strip():
        guidance = load_guidance(Path(args.combo_dir), pools)
    evaluator = AcetalEvaluator(pack_dir=pack, device=str(args.device))

    summary = []
    t0 = time.time()
    for r in range(1, int(args.runs) + 1):
        run_dir = out_dir / f"run_{r:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        rng = random.Random(int(args.seed) + r * 10007)

        df_eval, df_curve, meta, df_front = run_one(
            run_id=int(r),
            evaluator=evaluator,
            pools=pools,
            eval_budget=int(args.evals_per_run),
            pop_size=int(args.pop_size),
            cx_prob=float(args.cx_prob),
            mut_prob=float(args.mut_prob),
            rng=rng,
            batch_size=int(args.batch_size),
            logp_target=float(args.logp_target),
            guidance=guidance,
            guided_ratio=float(args.guided_ratio),
            iter_nohit_stop_config=iter_nohit_config_from_args(args),
        )

        df_eval.to_csv(run_dir / "evaluated.csv", index=False, encoding="utf-8-sig")
        df_curve.to_csv(run_dir / "best_curve.csv", index=False, encoding="utf-8-sig")
        df_front.to_csv(run_dir / "pareto_front_final.csv", index=False, encoding="utf-8-sig")
        (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        summary.append(
            {
                "run": int(r),
                "best_pred_LOGk2_mean_seeds": float(meta["best"]["pred_LOGk2_mean_seeds"]),
                "best_pred_LOGk2_std_seeds": float(meta["best"]["pred_LOGk2_std_seeds"]),
                "best_smiles": str(meta["best"]["smiles"]),
                "best_sa_score": float(meta["best"]["sa_score"]),
                "best_logP": float(meta["best"]["logP"]),
                "pareto_front_size": int(meta["pareto_front_size"]),
                "oracle_evals": int(meta.get("oracle_evals", 0)),
                "early_stopped": bool(meta.get("early_stopped", False)),
                "iter_nohit_early_stopped": bool(meta.get("iter_nohit_early_stopped", False)),
                "saved_oracle_evals": int(meta.get("saved_oracle_evals", 0)),
            }
        )

    pd.DataFrame(summary).sort_values("best_pred_LOGk2_mean_seeds", ascending=False).to_csv(
        out_dir / "summary_runs.csv", index=False, encoding="utf-8-sig"
    )
    meta_all = {
        "runs": int(args.runs),
        "evals_per_run": int(args.evals_per_run),
        "total_oracle_evals": int(args.runs) * int(args.evals_per_run),
        "elapsed_sec": float(time.time() - t0),
        "params": vars(args),
    }
    (out_dir / "meta_all.json").write_text(json.dumps(meta_all, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
