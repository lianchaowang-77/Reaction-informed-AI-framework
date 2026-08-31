"""
Monte Carlo Tree Search (MCTS) for high predicted logk2 acetals, using the same
51D feature pipeline and hard-constraint filter as GA.

Key semantics (matches GA setup):
- Search space: base+new substituent pools (screen_uncertainty_5seed/enrichment_analysis/pool_union_true_uncapped_20x4.csv)
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
  - best_curve.csv: best-so-far vs oracle eval index
  - meta.json: run config and summary stats
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from ga import AcetalEvaluator, Gene, _canon_sub
from guidance_simple import load_guidance, sample_guided_assignment, sample_guided_assignment_with_meta


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


def _sample_rollout_gene(
    pools: Dict[str, List[str]],
    chosen: Dict[str, str],
    rng: random.Random,
    guidance: Optional[Dict[str, List[dict]]] = None,
    guided_ratio: float = 0.30,
) -> Gene:
    if guidance and rng.random() < float(guided_ratio):
        a = sample_guided_assignment(pools, guidance, rng=rng)
        a.update(chosen)
        return Gene(r1=a["R1"], r2=a["R2"], r3=a["R3"], r4=a["R4"])
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
    sim_batch: int = 200,
    guidance: Optional[Dict[str, List[dict]]] = None,
    guided_ratio: float = 0.30,
    iter_nohit_stop_config: Optional[IterNoHitStopConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    # Root has untried actions = pool for R1
    root = Node(depth=0, parent=None, action=None, untried=list(pools["R1"]))

    seen_run: set[str] = set()
    rows = []
    curve = []

    oracle_evals = 0
    attempts = 0
    best = -1e100
    iter_nohit_stopper = IterNoHitStopper(iter_nohit_stop_config)
    stop_requested = False
    flush_iter = 0
    last_report_eval = 0

    def path_for_gene(gene: Gene) -> List[Node]:
        """Return/create a tree path consistent with a fully specified gene."""
        actions = [gene.r1, gene.r2, gene.r3, gene.r4]
        pos_order = ["R1", "R2", "R3", "R4"]
        node = root
        path = [node]
        for depth, action in enumerate(actions, start=1):
            child = node.children.get(action)
            if child is None:
                if action in node.untried:
                    node.untried.remove(action)
                child = Node(depth=depth, parent=node, action=action)
                if depth < 4:
                    child.untried = list(pools[pos_order[depth]])
                node.children[action] = child
            node = child
            path.append(node)
        return path

    def make_proposals(n: int, *, force_guided: bool) -> List[GuidedProposal]:
        nonlocal attempts
        proposals: List[GuidedProposal] = []
        for _ in range(int(n)):
            attempts += 1
            if attempts > int(eval_budget) * int(max_attempt_factor):
                raise RuntimeError("MCTS exceeded the maximum raw-proposal attempt budget")
            if force_guided:
                assignment, meta = sample_guided_assignment_with_meta(pools, guidance, rng=rng)
                if not bool(meta.get("is_guided", False)):
                    raise RuntimeError("guided quota requested but no R3-R4 guidance was loaded")
                gene = Gene(assignment["R1"], assignment["R2"], assignment["R3"], assignment["R4"])
                proposals.append(
                    GuidedProposal(
                        gene,
                        is_guided=True,
                        guide_id=str(meta.get("guide_id", "")),
                        guide_weight=float(meta.get("guide_weight", 0.0)),
                        payload=path_for_gene(gene),
                    )
                )
                continue

            node = root
            path = [node]
            while node.depth < 4 and (not node.untried) and node.children:
                node = _uct_select(node, c=float(uct_c), rng=rng)
                path.append(node)
            if node.depth < 4 and node.untried:
                action = node.untried.pop(rng.randrange(len(node.untried)))
                child_depth = node.depth + 1
                child = Node(depth=child_depth, parent=node, action=action)
                if child_depth < 4:
                    pos_order = ["R1", "R2", "R3", "R4"]
                    child.untried = list(pools[pos_order[child_depth]])
                node.children[action] = child
                node = child
                path.append(node)
            chosen = _node_path_to_partial(node)
            gene = _sample_rollout_gene(pools, chosen, rng=rng, guidance=None, guided_ratio=0.0)
            proposals.append(GuidedProposal(gene, payload=path))
        return proposals

    while oracle_evals < int(eval_budget) and not stop_requested:
        flush_iter += 1
        target_effective = next_batch_target(oracle_evals, int(eval_budget), int(sim_batch))
        target_guided, target_unguided = quota_counts(target_effective, guided_ratio)
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
        current_iter_rows: List[dict] = []
        for result, proposal in guided_pairs + unguided_pairs:
            reward = float(result.pred_LOGk2_mean_seeds)
            _backprop(list(proposal.payload), reward=reward)
            oracle_evals += 1
            best = max(best, reward)
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
                "pred_LOGk2_mean_seeds": reward,
                "pred_LOGk2_std_seeds": float(result.pred_LOGk2_std_seeds),
                "phase": f"mcts_iter_{flush_iter:03d}_{'guided' if proposal.is_guided else 'unguided'}",
                "is_guided": bool(proposal.is_guided),
                "guide_id": str(proposal.guide_id),
                "guide_weight": float(proposal.guide_weight),
            }
            rows.append(row)
            current_iter_rows.append(row)
            curve.append({"oracle_eval_idx": int(oracle_evals), "best_pred_LOGk2_mean_seeds": float(best)})
        if iter_nohit_stopper.observe_iteration(current_iter_rows, int(oracle_evals), flush_iter):
            stop_requested = True
        if oracle_evals - last_report_eval >= 500 or stop_requested:
            print(
                f"[MCTS] oracle_evals={oracle_evals} best={best:.4f} "
                f"nohit_iters={iter_nohit_stopper.consecutive_nohit_iters}",
                flush=True,
            )
            last_report_eval = int(oracle_evals)

    df_eval = pd.DataFrame(rows)
    df_curve = pd.DataFrame(curve)
    meta = {
        "eval_budget": int(eval_budget),
        "oracle_evals": int(oracle_evals),
        "attempts_total": int(attempts),
        "uct_c": float(uct_c),
        "effective_oracle_evals_per_iteration": int(sim_batch),
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
        default=str(Path("datasets") / "acetal" / "pack" / "MCTS_results" / "mcts_5000eval_10runs_seed3_molwtSemantic"),
        help="Output directory for runs.",
    )
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--evals_per_run", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--uct_c", type=float, default=2.0, help="UCT exploration constant.")
    ap.add_argument("--batch_size", type=int, default=2048, help="Evaluator internal batch size.")
    ap.add_argument("--sim_batch", type=int, default=200, help="How many simulations to evaluate per batch (speed).")
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
            uct_c=float(args.uct_c),
            rng=rng,
            batch_size=int(args.batch_size),
            sim_batch=int(args.sim_batch),
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
