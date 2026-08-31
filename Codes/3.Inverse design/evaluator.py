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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdchem

import batch_evaluator as ev


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


def _score_value(mean: float, std: float, *, lam: float, thr: float, alpha: float) -> float:
    """
    score = mean - lam*std + alpha*max(0, mean - thr)
    """
    return float(mean) - float(lam) * float(std) + float(alpha) * max(0.0, float(mean) - float(thr))


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
        require_novel: bool = False,
        require_feasible: bool = True,
        seen_run: Optional[set[str]] = None,
        batch_size: int = 4096,
        score_lam: float = 1.0,
        score_thr: float = 6.099,
        score_alpha: float = 2.0,
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
            score = _score_value(mean, std, lam=float(score_lam), thr=float(score_thr), alpha=float(score_alpha))
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


