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

from rdkit import RDLogger

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
    guidance: Dict[str, List[dict]] | None = None,
    guided_ratio: float = 0.30,
    iter_nohit_stop_config: Optional[IterNoHitStopConfig] = None,
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
    best = -1e100

    baseline = 0.0
    n_updates = 0
    iter_nohit_stopper = IterNoHitStopper(iter_nohit_stop_config)
    stop_requested = False
    last_report_eval = 0

    algorithm_iter = 0
    while oracle_evals < int(eval_budget) and not stop_requested:
        algorithm_iter += 1
        iter_rows: List[dict] = []
        target_effective = min(int(sample_batch), int(eval_budget) - int(oracle_evals))
        target_guided, target_unguided = quota_counts(target_effective, guided_ratio)
        probs = {p: _softmax(logits[p], temperature=float(temperature)) for p in pos_order}

        def make_proposals(n: int, *, force_guided: bool) -> List[GuidedProposal]:
            nonlocal attempts_batches
            attempts_batches += 1
            proposals: List[GuidedProposal] = []
            for _ in range(int(n)):
                if force_guided:
                    assignment, meta = sample_guided_assignment_with_meta(pools, guidance, rng=rng)
                    if not bool(meta.get("is_guided", False)):
                        raise RuntimeError("guided quota requested but no R3-R4 guidance was loaded")
                    pick = {p: actions[p].index(assignment[p]) for p in pos_order}
                    proposals.append(
                        GuidedProposal(
                            Gene(*(actions[p][pick[p]] for p in pos_order)),
                            is_guided=True,
                            guide_id=str(meta.get("guide_id", "")),
                            guide_weight=float(meta.get("guide_weight", 0.0)),
                            payload=pick,
                        )
                    )
                else:
                    pick = {p: _sample_categorical(probs[p], rng=rng) for p in pos_order}
                    proposals.append(
                        GuidedProposal(
                            Gene(*(actions[p][pick[p]] for p in pos_order)),
                            payload=pick,
                        )
                    )
            return proposals

        guided_pairs = evaluate_to_quota(
            evaluator=evaluator,
            target=target_guided,
            proposal_factory=lambda n: make_proposals(n, force_guided=True),
            seen_run=seen_run,
            batch_size=int(batch_size),
        )
        unguided_pairs = evaluate_to_quota(
            evaluator=evaluator,
            target=target_unguided,
            proposal_factory=lambda n: make_proposals(n, force_guided=False),
            seen_run=seen_run,
            batch_size=int(batch_size),
        )

        for r, proposal in guided_pairs + unguided_pairs:
            oracle_evals += 1
            reward = float(r.pred_LOGk2_mean_seeds)
            if n_updates == 0:
                baseline = reward
            else:
                baseline = float(baseline_beta) * float(baseline) + (1.0 - float(baseline_beta)) * reward
            adv = reward - float(baseline)
            pick = dict(proposal.payload)
            for p in pos_order:
                one = np.zeros_like(logits[p])
                one[pick[p]] = 1.0
                logits[p] += float(lr) * float(adv) * (one - probs[p])
                if float(entropy_coef) > 0:
                    uni = np.ones_like(probs[p]) / float(len(probs[p]))
                    logits[p] += float(entropy_coef) * (uni - probs[p])
            n_updates += 1

            best = max(best, reward)
            pos = evaluator.pos_cache.get(r.smiles, {})
            phase = f"rl_iter_{algorithm_iter:03d}_{'guided' if proposal.is_guided else 'unguided'}"
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
                "pred_LOGk2_mean_seeds": float(r.pred_LOGk2_mean_seeds),
                "pred_LOGk2_std_seeds": float(r.pred_LOGk2_std_seeds),
                "baseline": float(baseline),
                "advantage": float(adv),
                "phase": phase,
                "is_guided": bool(proposal.is_guided),
                "guide_id": str(proposal.guide_id),
                "guide_weight": float(proposal.guide_weight),
            }
            rows.append(row)
            iter_rows.append(row)
            curve.append({"oracle_eval_idx": int(oracle_evals), "best_pred_LOGk2_mean_seeds": float(best)})

        if iter_nohit_stopper.observe_iteration(iter_rows, int(oracle_evals), algorithm_iter):
            stop_requested = True
        if oracle_evals - last_report_eval >= 500 or stop_requested:
            print(
                f"[RL] oracle_evals={oracle_evals} best={best:.4f} "
                f"nohit_iters={iter_nohit_stopper.consecutive_nohit_iters}",
                flush=True,
            )
            last_report_eval = int(oracle_evals)

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
        "effective_oracle_evals_per_iteration": int(sample_batch),
        "guided_ratio_requested": float(guided_ratio),
        "guided_oracle_evals": int(sum(bool(row.get("is_guided", False)) for row in rows)),
        "unguided_oracle_evals": int(sum(not bool(row.get("is_guided", False)) for row in rows)),
        "effective_guided_oracle_ratio": float(sum(bool(row.get("is_guided", False)) for row in rows) / len(rows)),
        "best_pred_LOGk2_mean_seeds": float(best if best > -1e90 else float("nan")),
    }
    meta.update(iter_nohit_stopper.to_meta(eval_budget=int(eval_budget), actual_evals=int(oracle_evals)))
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
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--evals_per_run", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--entropy_coef", type=float, default=0.01)
    ap.add_argument("--baseline_beta", type=float, default=0.95, help="EMA factor for baseline (higher=slower).")
    ap.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature for action sampling.")
    ap.add_argument("--sample_batch", type=int, default=200, help="Number of candidates sampled per iteration (before filtering).")
    ap.add_argument("--batch_size", type=int, default=2048, help="Evaluator internal batch size.")
    ap.add_argument("--combo_dir", type=str, default="")
    ap.add_argument("--guided_ratio", type=float, default=0.30)
    add_iter_nohit_stop_args(ap)

    args = ap.parse_args()

    pack = Path(args.pack_dir)
    out_dir = Path(args.out_dir)
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
            guidance=guidance,
            guided_ratio=float(args.guided_ratio),
            iter_nohit_stop_config=iter_nohit_config_from_args(args),
        )

        df_eval.to_csv(run_dir / "evaluated.csv", index=False, encoding="utf-8-sig")
        df_curve.to_csv(run_dir / "best_curve.csv", index=False, encoding="utf-8-sig")
        (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        if not df_eval.empty:
            best_row = df_eval.sort_values("pred_LOGk2_mean_seeds", ascending=False).head(1).iloc[0].to_dict()
            summary.append(
                {
                    "run": int(r),
                    "best_pred_LOGk2_mean_seeds": float(best_row["pred_LOGk2_mean_seeds"]),
                    "best_pred_LOGk2_std_seeds": float(best_row["pred_LOGk2_std_seeds"]),
                    "best_smiles": str(best_row["smiles"]),
                    "best_sa_score": float(best_row["sa_score"]),
                    "best_logP": float(best_row["logP"]),
                    "oracle_evals": int(meta.get("oracle_evals", 0)),
                    "early_stopped": bool(meta.get("early_stopped", False)),
                    "iter_nohit_early_stopped": bool(meta.get("iter_nohit_early_stopped", False)),
                    "saved_oracle_evals": int(meta.get("saved_oracle_evals", 0)),
                }
            )
        else:
            summary.append({"run": int(r), "best_pred_LOGk2_mean_seeds": float("nan"), "best_smiles": "", "best_sa_score": float("nan"), "best_logP": float("nan"), "oracle_evals": int(meta.get("oracle_evals", 0))})

    df_sum = pd.DataFrame(summary).sort_values("best_pred_LOGk2_mean_seeds", ascending=False)
    df_sum.to_csv(out_dir / "summary_runs.csv", index=False, encoding="utf-8-sig")
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
