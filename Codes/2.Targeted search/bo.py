"""
Bayesian Optimization (BO) for high predicted logk2 acetals on a discrete combinatorial search space.

Design:
- Search space: base+new substituent pools for R1..R4 (categorical choices).
- Oracle: AcetalEvaluator (molwt-semantic 51D -> DNN3 5-seed ensemble mean/std), with hard constraints + novelty.
- Surrogate: RandomForestRegressor on one-hot encoding of categorical choices.
  Uncertainty estimated via per-tree variance.
- Acquisition: UCB = mu + kappa * sigma (maximize), evaluated on a random candidate set.

Why RF-UCB:
- No heavy GP dependencies, robust on small noisy datasets, uncertainty from tree ensemble.

Outputs (per run):
  - evaluated.csv
  - best_curve.csv
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
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import RDLogger
from sklearn.ensemble import RandomForestRegressor

from ga import AcetalEvaluator, Gene, _canon_sub
from guidance_simple import load_guidance, sample_guided_assignment_with_meta


def quota_counts(total: int, guided_ratio: float) -> Tuple[int, int]:
    """Return guided/unguided counts whose sum is exactly ``total``."""
    n = max(0, int(total))
    ratio = min(1.0, max(0.0, float(guided_ratio)))
    guided = int(round(n * ratio))
    return guided, n - guided


def budget_reached(actual_evals: int, eval_budget: int) -> bool:
    return int(actual_evals) >= int(eval_budget)


def next_batch_target(actual_evals: int, eval_budget: int, batch_target: int) -> int:
    remaining = max(0, int(eval_budget) - int(actual_evals))
    return min(max(0, int(batch_target)), remaining)


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
    return IterNoHitStopConfig(
        enabled=bool(getattr(args, "iter_nohit_stop", False)),
        hit_threshold=float(getattr(args, "iter_nohit_threshold", 6.1)),
        patience_iters=int(getattr(args, "iter_nohit_patience", 3)),
        min_evals=int(getattr(args, "iter_nohit_min_evals", 0)),
    )


class IterNoHitStopper:
    """Stop when consecutive algorithm iterations produce no hit above threshold."""

    def __init__(self, config: Optional[IterNoHitStopConfig]) -> None:
        self.config = config or IterNoHitStopConfig(enabled=False)
        self.iterations_checked = 0
        self.consecutive_nohit_iters = 0
        self.early_stopped = False
        self.stop_eval: Optional[int] = None
        self.stop_iteration: Optional[int] = None
        self.stop_reason = ""

    def observe_iteration(self, new_rows: List[Any], n_evals: int, iteration: int) -> bool:
        cfg = self.config
        if not cfg.enabled or self.early_stopped or int(n_evals) <= int(cfg.min_evals):
            return False
        self.iterations_checked += 1
        has_hit = any(
            (pred := _row_value(row, ["pred_LOGk2_mean_seeds", "pred_mean"], None)) is not None
            and float(pred) > float(cfg.hit_threshold)
            for row in new_rows
        )
        self.consecutive_nohit_iters = 0 if has_hit else self.consecutive_nohit_iters + 1
        if self.consecutive_nohit_iters < int(cfg.patience_iters):
            return False
        self.early_stopped = True
        self.stop_eval = int(n_evals)
        self.stop_iteration = int(iteration)
        self.stop_reason = (
            f"no pred_LOGk2_mean_seeds > {cfg.hit_threshold} in "
            f"{cfg.patience_iters} consecutive algorithm iterations"
        )
        return True

    def to_meta(self, eval_budget: int, actual_evals: int) -> Dict[str, Any]:
        cfg = self.config
        return {
            "iter_nohit_stop_enabled": bool(cfg.enabled),
            "iter_nohit_threshold": float(cfg.hit_threshold),
            "iter_nohit_patience": int(cfg.patience_iters),
            "iter_nohit_min_evals": int(cfg.min_evals),
            "iter_nohit_early_stopped": bool(self.early_stopped),
            "iter_nohit_stop_eval": self.stop_eval,
            "iter_nohit_stop_iteration": self.stop_iteration,
            "iter_nohit_stop_reason": str(self.stop_reason),
            "iter_nohit_iterations_checked": int(self.iterations_checked),
            "iter_nohit_consecutive_nohit_iters_at_end": int(self.consecutive_nohit_iters),
            "actual_oracle_evals": int(actual_evals),
            "saved_oracle_evals": max(0, int(eval_budget) - int(actual_evals)),
        }


RDLogger.DisableLog("rdApp.warning")


@dataclass(frozen=True)
class Proposal:
    gene: Gene
    is_guided: bool = False
    guide_id: str = ""
    guide_weight: float = 0.0


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


def _gene_to_indices(g: Gene, idx: Dict[str, Dict[str, int]]) -> Tuple[int, int, int, int]:
    return (idx["R1"][g.r1], idx["R2"][g.r2], idx["R3"][g.r3], idx["R4"][g.r4])


def _encode_onehot(idxs: List[Tuple[int, int, int, int]], sizes: Dict[str, int]) -> np.ndarray:
    # Concatenate one-hot blocks: R1 | R2 | R3 | R4
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
    # mean across trees, std across trees
    preds = np.stack([t.predict(X) for t in model.estimators_], axis=0)  # (n_trees, n)
    mu = np.mean(preds, axis=0)
    sigma = np.std(preds, axis=0, ddof=0)
    return mu, sigma


def _propose_random(
    pools: Dict[str, List[str]],
    n: int,
    rng: random.Random,
    guidance: Optional[Dict[str, List[dict]]] = None,
    guided_ratio: float = 0.30,
    force_guided: Optional[bool] = None,
) -> List[Proposal]:
    out = []
    for _ in range(int(n)):
        use_guidance = bool(guidance) and (
            bool(force_guided) if force_guided is not None else rng.random() < float(guided_ratio)
        )
        if use_guidance:
            a, meta = sample_guided_assignment_with_meta(pools, guidance, rng=rng)
            out.append(
                Proposal(
                    gene=Gene(a["R1"], a["R2"], a["R3"], a["R4"]),
                    is_guided=bool(meta.get("is_guided", False)),
                    guide_id=str(meta.get("guide_id", "")),
                    guide_weight=float(meta.get("guide_weight", 0.0)),
                )
            )
        else:
            out.append(
                Proposal(
                    gene=Gene(
                        rng.choice(pools["R1"]),
                        rng.choice(pools["R2"]),
                        rng.choice(pools["R3"]),
                        rng.choice(pools["R4"]),
                    )
                )
            )
    return out


def _unique_proposals(proposals: List[Proposal]) -> List[Proposal]:
    by_gene: Dict[Gene, Proposal] = {}
    order: List[Gene] = []
    for proposal in proposals:
        gene = proposal.gene
        if gene not in by_gene:
            by_gene[gene] = proposal
            order.append(gene)
        elif proposal.is_guided and not by_gene[gene].is_guided:
            by_gene[gene] = proposal
    return [by_gene[gene] for gene in order]


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
    guidance: Optional[Dict[str, List[dict]]] = None,
    guided_ratio: float = 0.30,
    iter_nohit_stop_config: Optional[IterNoHitStopConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    seen_run: set[str] = set()
    evaluated_rows: List[dict] = []
    best_curve: List[dict] = []
    iter_nohit_stopper = IterNoHitStopper(iter_nohit_stop_config)
    stop_requested = False
    best = -math.inf

    def record_results(
        results: list,
        proposal_lookup: Dict[Gene, Proposal],
        phase: str,
        iteration_rows: List[dict],
    ) -> int:
        nonlocal best, stop_requested
        added = 0
        for result in results:
            proposal = proposal_lookup.get(result.gene, Proposal(result.gene))
            pos = evaluator.pos_cache.get(result.smiles, {})
            row = {
                "gene_R1_smiles": result.gene.r1,
                "gene_R2_smiles": result.gene.r2,
                "gene_R3_smiles": result.gene.r3,
                "gene_R4_smiles": result.gene.r4,
                "R1_smiles": pos.get("R1_smiles", ""),
                "R2_smiles": pos.get("R2_smiles", ""),
                "R3_smiles": pos.get("R3_smiles", ""),
                "R4_smiles": pos.get("R4_smiles", ""),
                "smiles": result.smiles,
                "sa_score": float(result.sa_score),
                "logP": float(result.logP),
                "pred_LOGk2_mean_seeds": float(result.pred_LOGk2_mean_seeds),
                "pred_LOGk2_std_seeds": float(result.pred_LOGk2_std_seeds),
                "phase": phase,
                "is_guided": bool(proposal.is_guided),
                "guide_id": str(proposal.guide_id),
                "guide_weight": float(proposal.guide_weight),
            }
            evaluated_rows.append(row)
            iteration_rows.append(row)
            added += 1
            best = max(best, float(result.pred_LOGk2_mean_seeds))
            best_curve.append({"oracle_eval_idx": len(evaluated_rows), "best_pred_LOGk2_mean_seeds": float(best)})
            if budget_reached(len(evaluated_rows), int(eval_budget)):
                break
        return added

    def evaluate_group_to_quota(
        ordered: List[Proposal],
        target: int,
        *,
        force_guided: bool,
        phase: str,
        iteration_rows: List[dict],
    ) -> int:
        added = 0
        cursor = 0
        backfill_rounds = 0
        while added < int(target) and not budget_reached(len(evaluated_rows), int(eval_budget)) and not stop_requested:
            need = min(int(target) - added, int(eval_budget) - len(evaluated_rows), 512)
            batch_proposals = ordered[cursor : cursor + need]
            cursor += len(batch_proposals)
            if not batch_proposals:
                backfill_rounds += 1
                if backfill_rounds > 50:
                    break
                batch_proposals = _propose_random(
                    pools,
                    need,
                    rng=rng,
                    guidance=guidance,
                    guided_ratio=guided_ratio,
                    force_guided=force_guided,
                )
            lookup = {proposal.gene: proposal for proposal in batch_proposals}
            results = evaluator.evaluate_batch(
                [proposal.gene for proposal in batch_proposals],
                require_novel=True,
                require_feasible=True,
                seen_run=seen_run,
                batch_size=int(batch_size),
            )
            added += record_results(results, lookup, phase, iteration_rows)
        return added

    # index maps for encoding
    idx: Dict[str, Dict[str, int]] = {p: {s: i for i, s in enumerate(pools[p])} for p in ["R1", "R2", "R3", "R4"]}
    sizes = {p: len(pools[p]) for p in ["R1", "R2", "R3", "R4"]}

    # Initial evaluations use an exact guided/unguided oracle quota.
    init_target = min(int(n_init), int(eval_budget))
    init_guided, init_unguided = quota_counts(init_target, guided_ratio)
    init_rows: List[dict] = []
    evaluate_group_to_quota([], init_guided, force_guided=True, phase="init_guided", iteration_rows=init_rows)
    evaluate_group_to_quota([], init_unguided, force_guided=False, phase="init_unguided", iteration_rows=init_rows)

    if not evaluated_rows:
        raise RuntimeError("BO init failed: no feasible+novel evaluations; check constraints/pools.")

    # BO loop
    it = 0
    while not budget_reached(len(evaluated_rows), int(eval_budget)) and not stop_requested:
        it += 1

        # Fit surrogate on current evaluated set (use gene indices only; pred is target)
        genes_eval = [Gene(r["gene_R1_smiles"], r["gene_R2_smiles"], r["gene_R3_smiles"], r["gene_R4_smiles"]) for r in evaluated_rows]
        idxs_eval = [_gene_to_indices(g, idx) for g in genes_eval]
        X = _encode_onehot(idxs_eval, sizes=sizes)
        y = np.array([float(r["pred_LOGk2_mean_seeds"]) for r in evaluated_rows], dtype=np.float32)

        model = RandomForestRegressor(
            n_estimators=int(rf_trees),
            random_state=12345 + int(run_id) * 1000 + int(it),
            n_jobs=-1,
            max_depth=int(rf_max_depth) if int(rf_max_depth) > 0 else None,
            min_samples_leaf=1,
        )
        model.fit(X, y)

        # Candidate set for acquisition optimization
        n_guided_candidates, n_unguided_candidates = quota_counts(int(n_candidates), guided_ratio)
        cand_proposals = _unique_proposals(
            _propose_random(
                pools,
                n_guided_candidates,
                rng=rng,
                guidance=guidance,
                guided_ratio=guided_ratio,
                force_guided=True,
            )
            + _propose_random(
                pools,
                n_unguided_candidates,
                rng=rng,
                guidance=guidance,
                guided_ratio=guided_ratio,
                force_guided=False,
            )
        )
        cand_genes = [proposal.gene for proposal in cand_proposals]
        cand_idxs = [_gene_to_indices(g, idx) for g in cand_genes]
        Xc = _encode_onehot(cand_idxs, sizes=sizes)
        mu, sigma = _rf_predict_mu_sigma(model, Xc)
        acq = mu + float(kappa) * sigma
        order = np.argsort(-acq)  # descending
        guided_ordered = [cand_proposals[int(j)] for j in order if cand_proposals[int(j)].is_guided]
        unguided_ordered = [cand_proposals[int(j)] for j in order if not cand_proposals[int(j)].is_guided]

        # Reserve the requested fraction of actual oracle evaluations for
        # target-aware guided candidates, rather than only guiding the pool.
        target_new = next_batch_target(len(evaluated_rows), int(eval_budget), int(propose_batch))
        target_guided, target_unguided = quota_counts(target_new, guided_ratio)
        iter_rows: List[dict] = []
        evaluate_group_to_quota(
            guided_ordered,
            target_guided,
            force_guided=True,
            phase=f"bo_iter_{it:03d}_guided",
            iteration_rows=iter_rows,
        )
        evaluate_group_to_quota(
            unguided_ordered,
            target_unguided,
            force_guided=False,
            phase=f"bo_iter_{it:03d}_unguided",
            iteration_rows=iter_rows,
        )
        if iter_nohit_stopper.observe_iteration(iter_rows, len(evaluated_rows), it):
            stop_requested = True

    df_eval = pd.DataFrame(evaluated_rows)
    df_curve = pd.DataFrame(best_curve)
    best_row = df_eval.sort_values("pred_LOGk2_mean_seeds", ascending=False).head(1).to_dict(orient="records")[0]
    guided_mask = df_eval["is_guided"].astype(bool)
    hit_mask = pd.to_numeric(df_eval["pred_LOGk2_mean_seeds"], errors="coerce") > 6.1
    meta = {
        "run_id": int(run_id),
        "eval_budget": int(eval_budget),
        "n_init": int(n_init),
        "n_candidates": int(n_candidates),
        "propose_batch": int(propose_batch),
        "kappa": float(kappa),
        "rf_trees": int(rf_trees),
        "rf_max_depth": int(rf_max_depth),
        "attempted_unique_smiles": int(len(seen_run)),
        "oracle_evals": int(len(evaluated_rows)),
        "guided_ratio_requested": float(guided_ratio),
        "guided_oracle_evals": int(guided_mask.sum()),
        "unguided_oracle_evals": int((~guided_mask).sum()),
        "effective_guided_oracle_ratio": float(guided_mask.mean()),
        "guided_hits_gt6p1": int((guided_mask & hit_mask).sum()),
        "unguided_hits_gt6p1": int(((~guided_mask) & hit_mask).sum()),
        "best": best_row,
    }
    meta.update(iter_nohit_stopper.to_meta(eval_budget=int(eval_budget), actual_evals=len(evaluated_rows)))
    return df_eval, df_curve, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack_dir", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--runs", default=10, type=int)
    ap.add_argument("--evals_per_run", default=5000, type=int)
    ap.add_argument("--n_init", default=200, type=int, help="Initial oracle evaluations (feasible+novel only).")
    ap.add_argument("--n_candidates", default=20000, type=int, help="Random candidate set size per BO iteration.")
    ap.add_argument("--propose_batch", default=200, type=int, help="Target new oracle evaluations per BO iteration.")
    ap.add_argument("--kappa", default=2.0, type=float, help="UCB exploration coefficient.")
    ap.add_argument("--rf_trees", default=300, type=int)
    ap.add_argument("--rf_max_depth", default=18, type=int, help="0 means None.")
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

        df_eval, df_curve, meta = run_one(
            run_id=int(r),
            evaluator=evaluator,
            pools=pools,
            eval_budget=int(args.evals_per_run),
            n_init=int(args.n_init),
            n_candidates=int(args.n_candidates),
            propose_batch=int(args.propose_batch),
            kappa=float(args.kappa),
            rng=rng,
            batch_size=int(args.batch_size),
            rf_trees=int(args.rf_trees),
            rf_max_depth=int(args.rf_max_depth),
            guidance=guidance,
            guided_ratio=float(args.guided_ratio),
            iter_nohit_stop_config=iter_nohit_config_from_args(args),
        )

        df_eval.to_csv(run_dir / "evaluated.csv", index=False, encoding="utf-8-sig")
        df_curve.to_csv(run_dir / "best_curve.csv", index=False, encoding="utf-8-sig")
        (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        summary.append(
            {
                "run": int(r),
                "best_pred_LOGk2_mean_seeds": float(meta["best"]["pred_LOGk2_mean_seeds"]),
                "best_pred_LOGk2_std_seeds": float(meta["best"]["pred_LOGk2_std_seeds"]),
                "best_smiles": str(meta["best"]["smiles"]),
                "best_sa_score": float(meta["best"]["sa_score"]),
                "best_logP": float(meta["best"]["logP"]),
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
