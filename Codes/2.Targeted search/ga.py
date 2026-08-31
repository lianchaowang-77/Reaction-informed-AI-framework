"""
Genetic Algorithm (GA) search for high predicted logk2 acetals, using the project's 51D feature pipeline.

Design choices (for rigor + reproducibility):
1) Candidate representation is (R1_smiles, R2_smiles, R3_smiles, R4_smiles) using the project's substituent schema.
   This avoids unreliable reverse-decomposition from whole-molecule SMILES.
2) Whole-molecule SMILES is deterministically constructed using the same RDKit assembly core used in
   generate_enlarged_acetal_libraries.py: "[1*]OC([3*])([4*])O[2*]".
3) Hard constraints enforced before expensive model calls:
   - SA < 5
   - -1 < logP < 5
   - novelty vs known canonical smiles set (71w + 184) shipped in pack
4) Oracle is the 5-seed DNN3 ensemble, output column pred_LOGk2_mean_seeds + pred_LOGk2_std_seeds.
5) "eval budget" counts oracle calls (i.e., candidates that pass hard constraints and get pred_LOGk2_mean_seeds computed).

Outputs (per run):
  - evaluated.csv: all oracle-evaluated candidates (with genes, smiles, SA/logP, pred_LOGk2_mean_seeds, pred_LOGk2_std_seeds)
  - best_curve.csv: best-so-far vs eval index
  - meta.json: run config and summary stats
Plus overall:
  - summary_runs.csv: best per run and aggregate stats
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdchem

import batch_evaluator as ev
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


def evaluate_to_quota(
    *, evaluator: Any, target: int,
    proposal_factory: Callable[[int], List[GuidedProposal]],
    seen_run: set[str], batch_size: int, max_rounds: int = 1000,
) -> List[Tuple[Any, GuidedProposal]]:
    wanted = max(0, int(target))
    accepted: List[Tuple[Any, GuidedProposal]] = []
    rounds = 0
    while len(accepted) < wanted:
        rounds += 1
        if rounds > int(max_rounds):
            raise RuntimeError(f"unable to fill effective evaluation quota: {len(accepted)}/{wanted}")
        raw = list(proposal_factory(wanted - len(accepted)))
        unique: List[GuidedProposal] = []
        seen_keys = set()
        for proposal in raw:
            key = _gene_key(proposal.gene)
            if key not in seen_keys:
                seen_keys.add(key)
                unique.append(proposal)
        if not unique:
            continue
        lookup = {_gene_key(proposal.gene): proposal for proposal in unique}
        results = evaluator.evaluate_batch(
            [proposal.gene for proposal in unique], require_novel=True,
            require_feasible=True, seen_run=seen_run, batch_size=int(batch_size),
        )
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
    return IterNoHitStopConfig(
        enabled=bool(getattr(args, "iter_nohit_stop", False)),
        hit_threshold=float(getattr(args, "iter_nohit_threshold", 6.1)),
        patience_iters=int(getattr(args, "iter_nohit_patience", 3)),
        min_evals=int(getattr(args, "iter_nohit_min_evals", 0)),
    )


class IterNoHitStopper:
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
            and float(pred) > float(cfg.hit_threshold) for row in new_rows
        )
        self.consecutive_nohit_iters = 0 if has_hit else self.consecutive_nohit_iters + 1
        if self.consecutive_nohit_iters < int(cfg.patience_iters):
            return False
        self.early_stopped = True
        self.stop_eval, self.stop_iteration = int(n_evals), int(iteration)
        self.stop_reason = f"no pred_LOGk2_mean_seeds > {cfg.hit_threshold} in {cfg.patience_iters} consecutive algorithm iterations"
        return True

    def to_meta(self, eval_budget: int, actual_evals: int) -> Dict[str, Any]:
        cfg = self.config
        return {
            "iter_nohit_stop_enabled": bool(cfg.enabled), "iter_nohit_threshold": float(cfg.hit_threshold),
            "iter_nohit_patience": int(cfg.patience_iters), "iter_nohit_min_evals": int(cfg.min_evals),
            "iter_nohit_early_stopped": bool(self.early_stopped), "iter_nohit_stop_eval": self.stop_eval,
            "iter_nohit_stop_iteration": self.stop_iteration, "iter_nohit_stop_reason": str(self.stop_reason),
            "iter_nohit_iterations_checked": int(self.iterations_checked),
            "iter_nohit_consecutive_nohit_iters_at_end": int(self.consecutive_nohit_iters),
            "actual_oracle_evals": int(actual_evals),
            "saved_oracle_evals": max(0, int(eval_budget) - int(actual_evals)),
        }


R_COLS = ["R1_smiles", "R2_smiles", "R3_smiles", "R4_smiles"]


@dataclass(frozen=True)
class Gene:
    r1: str
    r2: str
    r3: str
    r4: str

    def as_dict(self) -> Dict[str, str]:
        return {"R1_smiles": self.r1, "R2_smiles": self.r2, "R3_smiles": self.r3, "R4_smiles": self.r4}


@dataclass
class EvalResult:
    gene: Gene
    smiles: str
    sa_score: float
    logP: float
    pred_LOGk2_mean_seeds: float
    pred_LOGk2_std_seeds: float
    score: float


def _score_value(mean: float, std: float) -> float:
    """
    In this no-reward variant, optimization target is directly pred_LOGk2_mean_seeds.
    std/bonus terms are intentionally disabled.
    """
    return float(mean)


def _canon_sub(s: str) -> Optional[str]:
    s0 = str(s).strip()
    if not s0:
        return None
    if s0 in {"H"}:
        s0 = "[H]"
    m = Chem.MolFromSmiles(s0)
    if m is None:
        return None
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=False)


@dataclass(frozen=True)
class SubPrep:
    mol: Chem.Mol
    dummy_idx: int
    neigh_idx: int


def _prepare_substituent(sub_smiles: str) -> Optional[SubPrep]:
    """
    Prepare a substituent (with exactly one dummy atom '*') for attachment.
    This is the same logic used in generate_enlarged_acetal_libraries.py.
    """
    s = str(sub_smiles).strip()
    if not s or s in {"[H]", "H"}:
        return None
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    dummies = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 1:
        return None
    d_idx = int(dummies[0])
    neis = [n.GetIdx() for n in m.GetAtomWithIdx(d_idx).GetNeighbors()]
    if len(neis) != 1:
        return None
    return SubPrep(mol=m, dummy_idx=d_idx, neigh_idx=int(neis[0]))


def _attach_one(core: Chem.Mol, label: int, sub_smiles: str, cache: Dict[str, Optional[SubPrep]]) -> Chem.Mol:
    """
    Attach substituent (with exactly one dummy atom) onto the core atom whose dummy has isotope==label.
    This is identical to generate_enlarged_acetal_libraries._attach_one().
    """
    if sub_smiles in {"[H]", "H", ""}:
        # remove corresponding core dummy, replace by H implicitly
        rw = Chem.RWMol(core)
        for a in core.GetAtoms():
            if a.GetAtomicNum() == 0 and int(a.GetIsotope()) == int(label):
                rw.RemoveAtom(a.GetIdx())
                out = rw.GetMol()
                Chem.SanitizeMol(out)
                return out
        return core

    prep = cache.get(sub_smiles)
    if prep is None:
        prep = _prepare_substituent(sub_smiles)
        cache[sub_smiles] = prep
    if prep is None:
        raise ValueError(f"bad substituent smiles: {sub_smiles}")

    # find dummy on core
    d_idx = None
    core_nei_idx = None
    for a in core.GetAtoms():
        if a.GetAtomicNum() == 0 and int(a.GetIsotope()) == int(label):
            d_idx = a.GetIdx()
            neis = [n.GetIdx() for n in a.GetNeighbors()]
            if len(neis) != 1:
                raise ValueError("core dummy must have exactly one neighbor")
            core_nei_idx = int(neis[0])
            break
    if d_idx is None or core_nei_idx is None:
        raise ValueError(f"cannot find core dummy label={label}")

    combo = Chem.CombineMols(core, prep.mol)
    rw = Chem.RWMol(combo)
    off = core.GetNumAtoms()
    rw.AddBond(core_nei_idx, off + prep.neigh_idx, rdchem.BondType.SINGLE)
    # remove dummy atoms: remove larger index first
    ridx = sorted([d_idx, off + prep.dummy_idx], reverse=True)
    for x in ridx:
        rw.RemoveAtom(int(x))
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out


def build_acetal_smiles(g: Gene, cache: Dict[str, Optional[SubPrep]]) -> Optional[str]:
    core = Chem.MolFromSmiles("[1*]OC([3*])([4*])O[2*]")
    if core is None:
        raise RuntimeError("failed to build core")
    try:
        m = core
        m = _attach_one(m, 1, g.r1, cache)  # R1-O-
        m = _attach_one(m, 2, g.r2, cache)  # -O-R2
        m = _attach_one(m, 3, g.r3, cache)  # direct group 1
        m = _attach_one(m, 4, g.r4, cache)  # direct group 2
        return Chem.MolToSmiles(m, canonical=True, isomericSmiles=False)
    except Exception:
        return None


class AcetalEvaluator:
    def __init__(self, pack_dir: Path, device: str = "cpu"):
        self.pack_dir = Path(pack_dir)
        ensemble_dir = self.pack_dir / "models" / "logk2_dnn3_5seed_ensemble"
        schema_csv = self.pack_dir / "schemas" / "x51d" / "feature_mapping_51d.csv"
        template = self.pack_dir / "schemas" / "x51d" / "virt_molwt_v1_full_X51D_from1500eprops_cdftFixed.csv.gz"
        cdft_csv = self.pack_dir / "props" / "cdft9_lookup" / "cdft6_lookup_merged.csv"
        known_csv_gz = self.pack_dir / "data" / "known_molecules" / "known_canonical_smiles_71w184.csv.gz"
        suite_root_txt = self.pack_dir / "models" / "eprops8_suite" / "ORIGINAL_SUITE_ROOT.txt"

        for p in [ensemble_dir, schema_csv, template, cdft_csv, known_csv_gz, suite_root_txt]:
            if not p.exists():
                raise FileNotFoundError(str(p))

        self.gids = ev._read_gids_from_template(template)
        schema = ev._load_51d_schema(schema_csv)
        self.specs = [schema[g] for g in self.gids]

        needed_cdft_props = sorted({s.prop for s in self.specs if s.block == "sub" and s.prop})
        self.cdft_lookup = ev._load_cdft_lookup(cdft_csv, needed_cdft_props)
        self.needed_eprops = [s.label for s in self.specs if s.block == "eprops"]

        self.known = ev._load_known_canonical_smiles_set(known_csv_gz)
        self.sascorer = ev._load_sascorer()
        self.norm_map = ev._load_norm_map(self.pack_dir / "data" / "substituents" / "base_new" / "substituent_normalization_map.csv")
        self.molwt_cache = ev._load_molwt_cache_csv(self.pack_dir / "data" / "substituents" / "base_new" / "substituent_molwt_cache.csv")

        member_dirs = sorted([p for p in Path(ensemble_dir).glob("member_*") if p.is_dir()])
        if not member_dirs:
            raise FileNotFoundError(f"no member_* under {ensemble_dir}")
        self.pred = ev.DNN3EnsemblePredictor(member_dirs=member_dirs, device=str(device))
        suite_root = self._resolve_eprops_suite_root(suite_root_txt)
        self.eprops = ev.EpropsSuitePredictor(suite_root=suite_root)

        # caches
        self.sa_logp_cache: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        self.eprops_cache: Dict[str, Dict[str, float]] = {}
        self.sub_desc_cache: Dict[str, Dict[str, float]] = {}
        self.cdft_cache: Dict[str, Dict[str, float]] = {}
        self.global_cache: Dict[str, Dict[str, float]] = {}
        self.pos_cache: Dict[str, Dict[str, str]] = {}  # canonical acetal smiles -> assigned positions (molwt rule)

    @staticmethod
    def _is_valid_eprops_suite_root(root: Path) -> bool:
        try:
            if not root.exists():
                return False
            if not (root / "suite_config.json").exists():
                return False
            # Expect at least one target subdir with config.json and keras model file.
            for sub in root.iterdir():
                if not sub.is_dir():
                    continue
                if not (sub / "config.json").exists():
                    continue
                if (sub / "electronic_model.keras").exists():
                    return True
                if any(p.name.startswith("DNN_") and p.suffix == ".keras" for p in sub.iterdir() if p.is_file()):
                    return True
                if len([p for p in sub.iterdir() if p.is_file() and p.suffix == ".keras"]) == 1:
                    return True
            return False
        except Exception:
            return False

    def _resolve_eprops_suite_root(self, suite_root_txt: Path) -> Path:
        txt = suite_root_txt.read_text(encoding="utf-8").replace("\ufeff", "").strip()
        candidates: List[Path] = []
        if txt:
            candidates.append(Path(txt))

        # Pack-local preferred candidates (for portable deployment)
        candidates.append(self.pack_dir / "models" / "eprops8_suite_full")
        candidates.append(self.pack_dir / "models" / "eprops8_suite")

        # PyInstaller runtime candidates
        try:
            exe_dir = Path(sys.executable).resolve().parent
            candidates.append(exe_dir / "eprops8_suite_full")
            candidates.append(exe_dir / "_internal" / "eprops8_suite_full")
        except Exception:
            pass
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass) / "eprops8_suite_full")

        seen = set()
        uniq: List[Path] = []
        for c in candidates:
            try:
                k = str(c.resolve())
            except Exception:
                k = str(c)
            if k in seen:
                continue
            seen.add(k)
            uniq.append(c)

        for c in uniq:
            if self._is_valid_eprops_suite_root(c):
                return c

        tried = "; ".join(str(x) for x in uniq[:8])
        raise FileNotFoundError(
            "Cannot locate a valid eprops8 suite root. Tried: " + tried
        )

    def evaluate_batch(
        self,
        genes: List[Gene],
        require_novel: bool = True,
        require_feasible: bool = True,
        seen_run: Optional[set[str]] = None,
        batch_size: int = 4096,
    ) -> List[EvalResult]:
        """
        Evaluate a list of genes and return only those that pass hard constraints and get pred_LOGk2_mean_seeds.
        """
        cache_attach: Dict[str, Optional[SubPrep]] = {}

        smiles_raw: List[Optional[str]] = []
        for g in genes:
            smiles_raw.append(build_acetal_smiles(g, cache_attach))

        can_list: List[Optional[str]] = []
        sa_list: List[Optional[float]] = []
        lp_list: List[Optional[float]] = []
        keep_idx: List[int] = []

        for i, smi in enumerate(smiles_raw):
            if not smi:
                can_list.append(None)
                sa_list.append(None)
                lp_list.append(None)
                continue
            can = ev._canon_mol_smiles(smi)
            if not can:
                can_list.append(None)
                sa_list.append(None)
                lp_list.append(None)
                continue

            can_list.append(can)
            if can in self.sa_logp_cache:
                sa, lp = self.sa_logp_cache[can]
            else:
                sa, lp = ev._calc_sa_logp(can, sascorer=self.sascorer)
                self.sa_logp_cache[can] = (sa, lp)
            sa_list.append(sa)
            lp_list.append(lp)

            if require_feasible:
                if sa is None or lp is None:
                    continue
                if not (sa < 5 and lp > -1 and lp < 5):
                    continue
            if require_novel:
                if can in self.known:
                    continue
                if seen_run is not None and can in seen_run:
                    continue

            # Assign R1..R4 by molwt rule from the whole-molecule SMILES (molwt standardized semantics).
            if can in self.pos_cache:
                pos = self.pos_cache[can]
            else:
                try:
                    pos = ev._assign_R1_R4_from_acetal_smiles_molwt(
                        can, molwt_cache=self.molwt_cache, norm_map=self.norm_map
                    )
                except Exception:
                    continue
                self.pos_cache[can] = pos

            # store assigned positions on the gene for later; we'll recompute in the build loop using cache
            # (kept separate to avoid changing the Gene dataclass).

            keep_idx.append(i)

        if not keep_idx:
            return []

        smiles_keep = [can_list[i] for i in keep_idx]
        assert all(s is not None for s in smiles_keep)
        smiles_keep2 = [str(s) for s in smiles_keep]

        # eprops8 prediction for keepers
        missing = [s for s in smiles_keep2 if s not in self.eprops_cache]
        if missing:
            ep_df = self.eprops.predict(missing, batch_size=int(batch_size))
            for _, r in ep_df.iterrows():
                cs = str(r["smiles"]).strip()
                self.eprops_cache[cs] = {t: float(r[t]) for t in self.needed_eprops}

        # build 51D vectors
        vecs: List[np.ndarray] = []
        out_rows: List[Tuple[int, str]] = []
        for i in keep_idx:
            can = str(can_list[i])
            g = genes[i]
            gene_pos = g.as_dict()
            pos = self.pos_cache.get(can)
            if pos is None:
                continue

            # per-position substituent desc + cdft
            sub_desc_by_pos = {}
            cdft_by_pos = {}
            ok = True
            for p in ["R1", "R2", "R3", "R4"]:
                unc = pos[f"{p}_smiles"]
                capped = ev._to_capped_smiles_canonical(unc)

                if capped in self.sub_desc_cache:
                    sub_desc = self.sub_desc_cache[capped]
                else:
                    m = Chem.MolFromSmiles(capped)
                    if m is None:
                        ok = False
                        break
                    sub_desc = ev._compute_sub_rdkit_descs(m)
                    self.sub_desc_cache[capped] = sub_desc
                sub_desc_by_pos[p] = sub_desc

                if capped in self.cdft_cache:
                    cd = self.cdft_cache[capped]
                else:
                    cd = self.cdft_lookup.get(capped) or {}
                    self.cdft_cache[capped] = cd
                cdft_by_pos[p] = cd

            if not ok:
                continue

            # global10
            if can in self.global_cache:
                gdesc = self.global_cache[can]
            else:
                m = Chem.MolFromSmiles(can)
                if m is None:
                    continue
                gdesc = ev._compute_global10_desc(m)
                self.global_cache[can] = gdesc

            ep = self.eprops_cache.get(can)
            if ep is None:
                continue

            vec = []
            ok2 = True
            for s in self.specs:
                if s.block == "rdkit":
                    v = float(sub_desc_by_pos[s.position][s.prop])
                elif s.block == "global":
                    v = float(gdesc[s.label])
                elif s.block == "sub":
                    cd = cdft_by_pos[s.position]
                    if s.prop not in cd:
                        ok2 = False
                        v = 0.0
                    else:
                        v = float(cd[s.prop])
                elif s.block == "eprops":
                    if s.label not in ep:
                        ok2 = False
                        v = 0.0
                    else:
                        v = float(ep[s.label])
                else:
                    ok2 = False
                    v = 0.0
                vec.append(v)
            if not ok2:
                continue

            vecs.append(np.asarray(vec, dtype=np.float32))
            out_rows.append((i, can))

        if not vecs:
            return []

        X = np.stack(vecs, axis=0)
        y, y_std = self.pred.predict_mean_std(X, batch_size=int(batch_size))
        y = np.asarray(y).reshape(-1)
        y_std = np.asarray(y_std).reshape(-1)

        results: List[EvalResult] = []
        for k, (i, can) in enumerate(out_rows):
            if seen_run is not None:
                seen_run.add(can)
            sa = float(sa_list[i]) if sa_list[i] is not None else float("nan")
            lp = float(lp_list[i]) if lp_list[i] is not None else float("nan")
            mean = float(y[k])
            std = float(y_std[k])
            score = _score_value(mean, std)
            results.append(
                EvalResult(
                    gene=genes[i],
                    smiles=can,
                    sa_score=sa,
                    logP=lp,
                    pred_LOGk2_mean_seeds=mean,
                    pred_LOGk2_std_seeds=std,
                    score=score,
                )
            )
        return results


def _tournament(pop: List[EvalResult], k: int, rng: random.Random) -> EvalResult:
    cand = rng.sample(pop, k=min(int(k), len(pop)))
    cand.sort(key=lambda r: r.pred_LOGk2_mean_seeds, reverse=True)
    return cand[0]


def _crossover(a: Gene, b: Gene, rng: random.Random) -> Tuple[Gene, Gene]:
    # uniform crossover on 4 loci
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


def run_one(
    run_id: int,
    evaluator: AcetalEvaluator,
    pools: Dict[str, List[str]],
    eval_budget: int,
    pop_size: int,
    elite_frac: float,
    tournament_k: int,
    cx_prob: float,
    mut_prob: float,
    rng: random.Random,
    batch_size: int,
    guidance: Optional[Dict[str, List[dict]]] = None,
    guided_ratio: float = 0.30,
    iter_nohit_stop_config: Optional[IterNoHitStopConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    seen_run: set[str] = set()
    evaluated: List[EvalResult] = []
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

    def propose_children(n: int, population: List[EvalResult]) -> List[GuidedProposal]:
        out: List[GuidedProposal] = []
        while len(out) < int(n):
            p1 = _tournament(population, tournament_k, rng).gene
            p2 = _tournament(population, tournament_k, rng).gene
            if rng.random() < float(cx_prob):
                c1, c2 = _crossover(p1, p2, rng)
            else:
                c1, c2 = p1, p2
            out.append(GuidedProposal(_mutate(c1, pools, mut_prob, rng)))
            if len(out) < int(n):
                out.append(GuidedProposal(_mutate(c2, pools, mut_prob, rng)))
        return out

    def record_pairs(
        pairs: List[Tuple[EvalResult, GuidedProposal]],
        *,
        phase: str,
    ) -> List[EvalResult]:
        nonlocal best, stop_requested
        recorded: List[EvalResult] = []
        for result, proposal in pairs:
            evaluated.append(result)
            recorded.append(result)
            annotations[result.smiles] = {
                "phase": phase,
                "is_guided": bool(proposal.is_guided),
                "guide_id": str(proposal.guide_id),
                "guide_weight": float(proposal.guide_weight),
            }
            best = max(best, float(result.pred_LOGk2_mean_seeds))
            best_curve.append({"eval_idx": len(evaluated), "best_pred_LOGk2_mean_seeds": best})
        return recorded

    # Initial population: exactly 30% guided effective evaluations.
    pop: List[EvalResult] = []
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

    # GA loop: generate children in batches
    elite_n = max(1, int(round(float(elite_frac) * float(pop_size))))
    gen = 0
    while len(evaluated) < int(eval_budget) and not stop_requested:
        gen += 1
        pop.sort(key=lambda r: r.pred_LOGk2_mean_seeds, reverse=True)
        elites = pop[:elite_n]

        target_new = next_batch_target(len(evaluated), int(eval_budget), int(pop_size))
        target_guided, target_unguided = quota_counts(target_new, guided_ratio)
        new_pop: List[EvalResult] = list(elites)
        gen_rows: List[EvalResult] = []
        guided_pairs = evaluate_to_quota(
            evaluator=evaluator,
            target=target_guided,
            proposal_factory=lambda n: propose_random(n, force_guided=True),
            seen_run=seen_run,
            batch_size=int(batch_size),
        )
        guided_results = record_pairs(guided_pairs, phase=f"ga_gen_{gen:03d}_guided")
        new_pop.extend(guided_results)
        gen_rows.extend(guided_results)
        unguided_pairs = evaluate_to_quota(
            evaluator=evaluator,
            target=target_unguided,
            proposal_factory=lambda n: propose_children(n, pop),
            seen_run=seen_run,
            batch_size=int(batch_size),
        )
        unguided_results = record_pairs(unguided_pairs, phase=f"ga_gen_{gen:03d}_unguided")
        new_pop.extend(unguided_results)
        gen_rows.extend(unguided_results)

        new_pop.sort(key=lambda r: r.pred_LOGk2_mean_seeds, reverse=True)
        pop = new_pop[: int(pop_size)]
        if iter_nohit_stopper.observe_iteration(gen_rows, len(evaluated), gen):
            stop_requested = True

    df_eval = pd.DataFrame(
        [
            {
                "gene_R1_smiles": r.gene.r1,
                "gene_R2_smiles": r.gene.r2,
                "gene_R3_smiles": r.gene.r3,
                "gene_R4_smiles": r.gene.r4,
                # Assigned positions by molwt rule (the schema used to build 51D and score).
                "R1_smiles": evaluator.pos_cache.get(r.smiles, {}).get("R1_smiles", ""),
                "R2_smiles": evaluator.pos_cache.get(r.smiles, {}).get("R2_smiles", ""),
                "R3_smiles": evaluator.pos_cache.get(r.smiles, {}).get("R3_smiles", ""),
                "R4_smiles": evaluator.pos_cache.get(r.smiles, {}).get("R4_smiles", ""),
                "smiles": r.smiles,
                "sa_score": r.sa_score,
                "logP": r.logP,
                "pred_LOGk2_mean_seeds": r.pred_LOGk2_mean_seeds,
                "pred_LOGk2_std_seeds": r.pred_LOGk2_std_seeds,
                "phase": annotations.get(r.smiles, {}).get("phase", ""),
                "is_guided": bool(annotations.get(r.smiles, {}).get("is_guided", False)),
                "guide_id": annotations.get(r.smiles, {}).get("guide_id", ""),
                "guide_weight": float(annotations.get(r.smiles, {}).get("guide_weight", 0.0)),
            }
            for r in evaluated
        ]
    )
    df_curve = pd.DataFrame(best_curve)
    best_row = df_eval.sort_values("pred_LOGk2_mean_seeds", ascending=False).head(1).to_dict(orient="records")[0]
    meta = {
        "run_id": int(run_id),
        "eval_budget": int(eval_budget),
        "oracle_evals": int(len(evaluated)),
        "attempted_unique_smiles": int(len(seen_run)),
        "effective_evals_per_iteration": int(pop_size),
        "guided_ratio_requested": float(guided_ratio),
        "guided_oracle_evals": int(sum(bool(annotations.get(r.smiles, {}).get("is_guided", False)) for r in evaluated)),
        "best": best_row,
    }
    meta["unguided_oracle_evals"] = int(len(evaluated) - int(meta["guided_oracle_evals"]))
    meta["effective_guided_oracle_ratio"] = float(meta["guided_oracle_evals"] / len(evaluated))
    meta.update(iter_nohit_stopper.to_meta(eval_budget=int(eval_budget), actual_evals=len(evaluated)))
    return df_eval, df_curve, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack_dir", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--runs", default=10, type=int)
    ap.add_argument("--evals_per_run", default=5000, type=int)
    ap.add_argument("--pop_size", default=200, type=int)
    ap.add_argument("--elite_frac", default=0.10, type=float)
    ap.add_argument("--tournament_k", default=3, type=int)
    ap.add_argument("--cx_prob", default=0.60, type=float)
    ap.add_argument("--mut_prob", default=0.25, type=float)
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

    # load pools (base+new union)
    pool_csv = Path(__file__).resolve().parent / "pool_union_simplify_from_enrichment_practical.csv"
    if not pool_csv.exists():
        raise FileNotFoundError(str(pool_csv))
    pdf = pd.read_csv(pool_csv, encoding="utf-8-sig")
    pools: Dict[str, List[str]] = {}
    for pos in ["R1", "R2", "R3", "R4"]:
        arr = [str(x).strip() for x in pdf[pdf["position"] == pos]["substituent"].tolist()]
        arr = [x for x in arr if x and x != "[H]"]
        # canonicalize and keep stable order
        seen = set()
        out = []
        for s in arr:
            c = _canon_sub(s)
            if not c:
                continue
            if c in seen:
                continue
            seen.add(c)
            out.append(c)
        if not out:
            raise RuntimeError(f"empty pool for {pos}")
        pools[pos] = out

    guidance = None
    if str(args.combo_dir).strip():
        guidance = load_guidance(Path(args.combo_dir), pools)

    # evaluator (shared across runs to reuse caches)
    evaluator = AcetalEvaluator(pack_dir=pack, device=str(args.device))

    summary = []
    t0 = time.time()
    for r in range(1, int(args.runs) + 1):
        run_dir = out_dir / f"run_{r:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        rng = random.Random(int(args.seed) + r * 10007)

        df_eval, df_curve, meta = run_one(
            run_id=r,
            evaluator=evaluator,
            pools=pools,
            eval_budget=int(args.evals_per_run),
            pop_size=int(args.pop_size),
            elite_frac=float(args.elite_frac),
            tournament_k=int(args.tournament_k),
            cx_prob=float(args.cx_prob),
            mut_prob=float(args.mut_prob),
            rng=rng,
            batch_size=int(args.batch_size),
            guidance=guidance,
            guided_ratio=float(args.guided_ratio),
            iter_nohit_stop_config=iter_nohit_config_from_args(args),
        )

        df_eval.to_csv(run_dir / "evaluated.csv", index=False, encoding="utf-8-sig")
        df_curve.to_csv(run_dir / "best_curve.csv", index=False, encoding="utf-8-sig")
        (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        best = df_eval.sort_values("pred_LOGk2_mean_seeds", ascending=False).head(1).iloc[0].to_dict()
        summary.append(
            {
                "run": int(r),
                "best_pred_LOGk2_mean_seeds": float(best["pred_LOGk2_mean_seeds"]),
                "best_pred_LOGk2_std_seeds": float(best["pred_LOGk2_std_seeds"]),
                "best_smiles": str(best["smiles"]),
                "best_sa_score": float(best["sa_score"]),
                "best_logP": float(best["logP"]),
                "oracle_evals": int(meta.get("oracle_evals", 0)),
                "early_stopped": bool(meta.get("early_stopped", False)),
                "iter_nohit_early_stopped": bool(meta.get("iter_nohit_early_stopped", False)),
                "saved_oracle_evals": int(meta.get("saved_oracle_evals", 0)),
            }
        )

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
