"""
Batch evaluator for acetal candidates:
  input (preferred): acetal SMILES
  optional: also accept pre-extracted substituents (uncapped, with '*' or '[H]')

What this evaluator does (per molecule):
1) Parse SMILES -> RDKit Mol
2) Compute SA score and logP (RDKit); apply hard constraints:
     SA < 5 and -1 < logP < 5
3) Check novelty against known canonical SMILES set (71w + 184)
4) Decompose molecule into 4 substituent fragments around the acetal center
5) Assign R1-R4 by molwt rule (as specified in 项目说明-新版.txt)
6) Build 51D features (RDKit24 + global10 + cdft9 lookup + eprops8 predicted)
7) Predict LOGk2 using a legacy DNN3.pt model (torch) with saved X/Y scalers

Outputs:
- CSV with molecule,smiles,pred_logk2,sa_score,logP,novel,is_feasible,... (and optionally R1..R4)

Performance notes:
- Designed for "optimization budget" scale (thousands to tens of thousands candidates).
- Uses caching for substituent RDKit desc + cdft and for per-molecule eprops + global desc.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors


try:  # pragma: no cover
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None

try:  # pragma: no cover
    import tensorflow as tf  # type: ignore
except Exception:  # pragma: no cover
    tf = None


def _require_torch() -> None:
    if torch is None or nn is None:  # pragma: no cover
        raise ModuleNotFoundError("torch not available; required for DNN3.pt inference")


def _require_tf() -> None:
    if tf is None:  # pragma: no cover
        raise ModuleNotFoundError("tensorflow not available; required for eprops8 suite inference")


def _load_sascorer():
    """
    Prefer RDKit contrib SA_Score, then bundled SA_Score, fallback to `sascorer`
    on PYTHONPATH.
    """
    import sys
    import importlib.util

    def _try_load_from_dir(sa_dir: Path):
        sa_py = sa_dir / "sascorer.py"
        fp_pkl = sa_dir / "fpscores.pkl.gz"
        if not sa_py.exists() or not fp_pkl.exists():
            return None
        spec = importlib.util.spec_from_file_location("sascorer", str(sa_py))
        if not spec or not spec.loader:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        return mod

    candidates: List[Path] = []

    # 1) RDKit contrib (source Python environment)
    try:
        from rdkit import RDConfig  # type: ignore

        candidates.append(Path(RDConfig.RDContribDir) / "SA_Score")
    except Exception:
        pass

    # 2) Bundled locations for PyInstaller one-dir/one-file runtime
    try:
        candidates.append(Path(__file__).resolve().parent / "SA_Score")
    except Exception:
        pass
    try:
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir / "SA_Score")
        candidates.append(exe_dir / "_internal" / "SA_Score")
    except Exception:
        pass
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "SA_Score")

    seen = set()
    for d in candidates:
        try:
            k = str(d.resolve())
        except Exception:
            k = str(d)
        if k in seen:
            continue
        seen.add(k)
        try:
            m = _try_load_from_dir(d)
            if m is not None:
                return m
        except Exception:
            continue

    # 3) Plain import fallback
    import sascorer  # type: ignore

    return sascorer


def _strip_dummy_isotopes(m: Chem.Mol) -> Chem.Mol:
    mm = Chem.Mol(m)
    for a in mm.GetAtoms():
        if a.GetAtomicNum() == 0:
            a.SetIsotope(0)
    return mm


def _canon_mol_smiles(s: str) -> Optional[str]:
    t = str(s).strip()
    if not t:
        return None
    m = Chem.MolFromSmiles(t)
    if m is None:
        return None
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=False)


def _hetero_atom_count(mol: Chem.Mol) -> float:
    atoms = mol.GetAtoms()
    return float(sum(1 for a in atoms if a.GetAtomicNum() not in (1, 6)))


def _halogen_count(mol: Chem.Mol) -> float:
    atoms = mol.GetAtoms()
    return float(sum(1 for a in atoms if a.GetSymbol() in ("F", "Cl", "Br", "I")))


def _compute_sub_rdkit_descs(mol: Chem.Mol) -> Dict[str, float]:
    return {
        "MolWt": float(Descriptors.MolWt(mol)),
        "HeavyAtomCount": float(Descriptors.HeavyAtomCount(mol)),
        "NumHAcceptors": float(Descriptors.NumHAcceptors(mol)),
        "NumHDonors": float(Descriptors.NumHDonors(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "MolLogP": float(Descriptors.MolLogP(mol)),
        "RingCount": float(Descriptors.RingCount(mol)),
        "NumAromaticRings": float(Descriptors.NumAromaticRings(mol)),
        "FractionCSP3": float(Descriptors.FractionCSP3(mol)),
        "HeteroAtomCount": _hetero_atom_count(mol),
        "HalogenCount": _halogen_count(mol),
    }


def _compute_global10_desc(mol: Chem.Mol) -> Dict[str, float]:
    atoms = mol.GetAtoms()
    hetero = sum(1 for a in atoms if a.GetAtomicNum() not in (1, 6))
    halo = sum(1 for a in atoms if a.GetAtomicNum() in (9, 17, 35, 53))
    formal_charge = sum(int(a.GetFormalCharge()) for a in atoms)
    return {
        "g_MolWt": float(Descriptors.MolWt(mol)),
        "g_HeavyAtomCount": float(Descriptors.HeavyAtomCount(mol)),
        "g_HeteroAtomCount": float(hetero),
        "g_HalogenCount": float(halo),
        "g_FormalCharge": float(formal_charge),
        "g_NumValenceElectrons": float(Descriptors.NumValenceElectrons(mol)),
        "g_NumRadicalElectrons": float(Descriptors.NumRadicalElectrons(mol)),
        "g_NumHAcceptors": float(Lipinski.NumHAcceptors(mol)),
        "g_NumHDonors": float(Lipinski.NumHDonors(mol)),
        "g_TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
    }


def _to_capped_smiles_canonical(uncapped: str) -> str:
    s = str(uncapped).strip()
    if s in ("", "nan", "None"):
        raise ValueError("empty substituent smiles")
    if s in ("[H]", "H"):
        return "C"

    mol = Chem.MolFromSmiles(s)
    if mol is None:
        raise ValueError(f"invalid substituent smiles: {s!r}")

    dummy_idxs = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if not dummy_idxs:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    if len(dummy_idxs) != 1:
        raise ValueError(f"expected exactly one '*' dummy atom, got {len(dummy_idxs)} in {s!r}")

    d_idx = dummy_idxs[0]
    neigh = list(mol.GetAtomWithIdx(d_idx).GetNeighbors())
    if len(neigh) != 1:
        raise ValueError(f"dummy atom must have 1 neighbor, got {len(neigh)} in {s!r}")
    n_idx = neigh[0].GetIdx()

    rw = Chem.RWMol(mol)
    rw.RemoveAtom(d_idx)
    if d_idx < n_idx:
        n_idx -= 1

    c_idx = rw.AddAtom(Chem.Atom("C"))
    rw.AddBond(n_idx, c_idx, Chem.BondType.SINGLE)
    capped = rw.GetMol()
    Chem.SanitizeMol(capped)
    capped = Chem.RemoveHs(capped)
    return Chem.MolToSmiles(capped, canonical=True, isomericSmiles=False)


def _attach_atom_symbol(uncapped: str) -> str:
    s = str(uncapped).strip()
    if s in {"[H]", "H"}:
        return "H"
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return ""
    dummy = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummy) != 1:
        return ""
    neigh = list(dummy[0].GetNeighbors())
    if len(neigh) != 1:
        return ""
    return str(neigh[0].GetSymbol())


def _calc_sa_logp(smiles: str, *, sascorer) -> Tuple[Optional[float], Optional[float]]:
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            return None, None
        sa = float(sascorer.calculateScore(mol))
        logp = float(Crippen.MolLogP(mol))
        return sa, logp
    except Exception:
        return None, None


def _read_gids_from_template(template_csv_gz: Path) -> List[str]:
    with gzip.open(template_csv_gz, "rt", encoding="utf-8-sig") as f:
        header = f.readline().strip().split(",")
    if len(header) < 3 or header[0].lower() != "molecule" or header[1].lower() != "smiles":
        raise ValueError("template header must start with molecule,smiles")
    gids = [c.strip() for c in header[2:] if c.strip()]
    if len(gids) != 51:
        raise ValueError(f"expected 51 gids, got {len(gids)}")
    return gids


@dataclass(frozen=True)
class ColSpec:
    gid: str
    block: str
    position: str
    prop: str
    label: str


def _load_51d_schema(schema_csv: Path) -> Dict[str, ColSpec]:
    df = pd.read_csv(schema_csv, encoding="utf-8-sig")
    need = {"global_id", "block", "position", "prop", "label"}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"schema_csv missing columns: {sorted(need - set(df.columns))}")
    out: Dict[str, ColSpec] = {}
    for _, r in df.iterrows():
        gid = str(int(r["global_id"]))
        out[gid] = ColSpec(
            gid=gid,
            block=str(r["block"]).strip(),
            position=("" if pd.isna(r["position"]) else str(r["position"]).strip()),
            prop=("" if pd.isna(r["prop"]) else str(r["prop"]).strip()),
            label=str(r["label"]).strip(),
        )
    return out


def _load_known_canonical_smiles_set(path_csv_gz: Path) -> set[str]:
    known = set()
    with gzip.open(path_csv_gz, "rt", encoding="utf-8") as f:
        header = f.readline()
        if "canonical_smiles" not in header:
            raise ValueError("known canonical smiles file must have header canonical_smiles")
        for line in f:
            s = line.strip()
            if s:
                known.add(s)
    return known


def _load_cdft_lookup(cdft_csv: Path, needed_props: Iterable[str]) -> Dict[str, Dict[str, float]]:
    df = pd.read_csv(cdft_csv, encoding="utf-8-sig")
    if "R_smiles" not in df.columns:
        raise ValueError("cdft csv must contain R_smiles (methyl-capped canonical)")
    need = ["R_smiles"] + list(needed_props)
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"cdft csv missing columns: {missing}")
    df = df.copy()
    df["R_smiles"] = df["R_smiles"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["R_smiles"], keep="first")
    m: Dict[str, Dict[str, float]] = {}
    for _, r in df.iterrows():
        key = str(r["R_smiles"]).strip()
        if not key:
            continue
        rec = {}
        for p in needed_props:
            rec[p] = float(pd.to_numeric(r[p], errors="coerce"))
        m[key] = rec
    return m


def _load_norm_map(norm_map_csv: Path) -> Dict[str, str]:
    """
    Best-effort loader for a substituent normalization map (raw -> normalized).
    If schema is unknown or file missing, returns {} (identity fallback).
    """
    if not norm_map_csv.exists():
        return {}
    df = pd.read_csv(norm_map_csv, encoding="utf-8-sig")
    # Try a few likely column pairs
    cand_pairs = [
        ("substituent_raw", "substituent_norm"),
        ("raw", "normalized"),
        ("raw_smiles", "normalized_smiles"),
        ("substituent_raw_smiles", "substituent"),
    ]
    raw_c = None
    norm_c = None
    lower = {str(c).lower(): c for c in df.columns}
    for a, b in cand_pairs:
        if a.lower() in lower and b.lower() in lower:
            raw_c = lower[a.lower()]
            norm_c = lower[b.lower()]
            break
    if raw_c is None or norm_c is None:
        return {}
    m = {}
    for _, r in df.iterrows():
        a = str(r[raw_c]).strip()
        b = str(r[norm_c]).strip()
        if a and b:
            m[a] = b
    return m


class TorchMLP(nn.Module):  # pragma: no cover
    def __init__(self, input_dim: int, hidden: List[int]):
        super().__init__()
        layers: List[nn.Module] = []
        d = int(input_dim)
        for u in hidden:
            layers.append(nn.Linear(d, int(u)))
            layers.append(nn.ReLU())
            d = int(u)
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DNN3Seed3Predictor:  # pragma: no cover
    def __init__(self, model_dir: Path, device: str = "cpu"):
        _require_torch()
        self.model_dir = model_dir
        self.device = str(device)
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        arch = cfg.get("arch", [128, 64, 32, 16])
        hidden = [int(x) for x in arch]

        with (model_dir / "Xdesc_scaler.pkl").open("rb") as f:
            self.x_scaler = pickle.load(f)
        with (model_dir / "Yscaler.pkl").open("rb") as f:
            self.y_scaler = pickle.load(f)

        self.model = TorchMLP(input_dim=51, hidden=hidden).to(torch.device(self.device))
        state = torch.load(str(model_dir / "DNN3.pt"), map_location=torch.device(self.device))
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, X_raw_51: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        _require_torch()
        Xs = self.x_scaler.transform(X_raw_51).astype(np.float32, copy=False)
        out = []
        n = int(Xs.shape[0])
        with torch.no_grad():
            for i in range(0, n, int(batch_size)):
                xb = torch.tensor(Xs[i : i + int(batch_size)], dtype=torch.float32, device=torch.device(self.device))
                yb = self.model(xb).detach().cpu().numpy().reshape(-1)
                out.append(yb)
        y_scaled = np.concatenate(out, axis=0).reshape(-1, 1)
        y = self.y_scaler.inverse_transform(y_scaled).reshape(-1)
        return y.astype(np.float32, copy=False)


class DNN3EnsemblePredictor:  # pragma: no cover
    def __init__(self, member_dirs: List[Path], device: str = "cpu"):
        _require_torch()
        if not member_dirs:
            raise ValueError("member_dirs is empty")
        self.device = str(device)
        self.members = []
        self.x_scaler = None
        self.y_scaler = None

        for md in member_dirs:
            cfg = json.loads((md / "config.json").read_text(encoding="utf-8"))
            arch = cfg.get("arch", [128, 64, 32, 16])
            hidden = [int(x) for x in arch]
            with (md / "Xdesc_scaler.pkl").open("rb") as f:
                x_scaler = pickle.load(f)
            with (md / "Yscaler.pkl").open("rb") as f:
                y_scaler = pickle.load(f)

            if self.x_scaler is None:
                self.x_scaler = x_scaler
            if self.y_scaler is None:
                self.y_scaler = y_scaler

            model = TorchMLP(input_dim=51, hidden=hidden).to(torch.device(self.device))
            state = torch.load(str(md / "DNN3.pt"), map_location=torch.device(self.device))
            model.load_state_dict(state)
            model.eval()
            self.members.append(model)

        if self.x_scaler is None or self.y_scaler is None:
            raise RuntimeError("failed to load scalers from members")

    def predict_mean_std(self, X_raw_51: np.ndarray, batch_size: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
        _require_torch()
        Xs = self.x_scaler.transform(X_raw_51).astype(np.float32, copy=False)
        preds = []
        n = int(Xs.shape[0])
        for model in self.members:
            out = []
            with torch.no_grad():
                for i in range(0, n, int(batch_size)):
                    xb = torch.tensor(Xs[i : i + int(batch_size)], dtype=torch.float32, device=torch.device(self.device))
                    yb = model(xb).detach().cpu().numpy().reshape(-1)
                    out.append(yb)
            y_scaled = np.concatenate(out, axis=0).reshape(-1, 1)
            y = self.y_scaler.inverse_transform(y_scaled).reshape(-1)
            preds.append(y.astype(np.float32, copy=False))
        P = np.stack(preds, axis=0)
        mean = P.mean(axis=0)
        std = P.std(axis=0, ddof=0)
        return mean, std


class EpropsSuitePredictor:  # pragma: no cover
    def __init__(self, suite_root: Path):
        _require_tf()
        self.suite_root = suite_root
        cfg = json.loads((suite_root / "suite_config.json").read_text(encoding="utf-8"))
        self.targets = [str(t) for t in cfg.get("targets", [])]
        fp = cfg.get("fingerprint", {})
        self.radius = int(fp.get("radius", 2))
        self.n_bits = int(fp.get("n_bits", 1024))
        self.use_chirality = bool(fp.get("use_chirality", False))

        self.bundle = []
        for t in self.targets:
            found = None
            for sub in suite_root.iterdir():
                if not sub.is_dir():
                    continue
                cpath = sub / "config.json"
                if not cpath.exists():
                    continue
                try:
                    c = json.loads(cpath.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if str(c.get("target", "")) == str(t):
                    found = sub
                    break
            if found is None:
                raise FileNotFoundError(f"Cannot find model dir for target '{t}' under {suite_root}")
            models = self._load_models_for_target(found)
            with (found / "Y_scaler.pkl").open("rb") as f:
                y_scaler = pickle.load(f)
            self.bundle.append({"target": t, "model_dir": str(found), "models": models, "y_scaler": y_scaler})

    @staticmethod
    def _load_models_for_target(tdir: Path):
        _require_tf()
        # Most single-task dirs store electronic_model.keras
        cand = tdir / "electronic_model.keras"
        if cand.exists():
            return [tf.keras.models.load_model(str(cand), compile=False)]
        # Fallback: DNN_*.keras ensemble
        files = sorted([p for p in tdir.iterdir() if p.is_file() and p.name.startswith("DNN_") and p.name.endswith(".keras")])
        if files:
            return [tf.keras.models.load_model(str(p), compile=False) for p in files]
        # Fallback: any single *.keras
        any_keras = sorted([p for p in tdir.iterdir() if p.is_file() and p.name.endswith(".keras")])
        if len(any_keras) == 1:
            return [tf.keras.models.load_model(str(any_keras[0]), compile=False)]
        raise FileNotFoundError(f"No keras model found under {tdir}")

    def _featurize_hashed_morgan_counts(self, smiles_list: List[str]) -> np.ndarray:
        from rdkit.Chem import AllChem

        X = np.zeros((len(smiles_list), int(self.n_bits)), dtype=np.float32)
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                continue
            fp = AllChem.GetHashedMorganFingerprint(mol, int(self.radius), nBits=int(self.n_bits), useChirality=bool(self.use_chirality))
            for k, v in fp.GetNonzeroElements().items():
                X[i, int(k)] = float(v)
        return X

    def predict(self, smiles_list: List[str], batch_size: int = 4096) -> pd.DataFrame:
        X = self._featurize_hashed_morgan_counts(smiles_list)
        out = pd.DataFrame({"smiles": list(smiles_list)})
        for item in self.bundle:
            preds = []
            for m in item["models"]:
                preds.append(m.predict(X, batch_size=int(batch_size), verbose=0).reshape(-1, 1))
            yhat_s = np.mean(np.stack(preds, axis=0), axis=0)
            yhat = item["y_scaler"].inverse_transform(yhat_s).reshape(-1)
            out[str(item["target"])] = yhat
        return out


def _find_acetal_center(mol: Chem.Mol) -> int:
    """
    Find the acetal center carbon for the project's acetal motif.

    Practical rule:
    - choose a carbon atom with exactly 2 oxygen neighbors and degree 4 if possible;
      fallback to any carbon with >=2 oxygen neighbors.
    """
    best = None
    best_key = None
    for a in mol.GetAtoms():
        if a.GetAtomicNum() != 6:
            continue
        # Acetal center is a saturated carbon (4 neighbors including implicit H).
        if int(a.GetTotalDegree()) != 4:
            continue
        if a.GetIsAromatic():
            continue
        # Only count single-bond oxygens: the project rule talks about "single-bond oxygen" connections.
        o_nei = []
        for n in a.GetNeighbors():
            if n.GetAtomicNum() != 8:
                continue
            b = mol.GetBondBetweenAtoms(a.GetIdx(), n.GetIdx())
            if b is not None and b.GetBondType() == Chem.rdchem.BondType.SINGLE:
                o_nei.append(n)

        if len(o_nei) < 2:
            continue

        # Prefer the carbon with the most single-bond oxygen neighbors (handles 2/3/4 O cases).
        # Tie-breakers: higher total degree, then lower atom index for determinism.
        key = (len(o_nei), int(a.GetTotalDegree()), -int(a.GetIdx()))
        if best_key is None or key > best_key:
            best_key = key
            best = a.GetIdx()
    if best is None:
        raise ValueError("cannot find acetal center (no carbon with >=2 O neighbors)")
    return int(best)


def _clear_atom_mapnums(m: Chem.Mol) -> Chem.Mol:
    rw = Chem.RWMol(m)
    for a in rw.GetAtoms():
        a.SetAtomMapNum(0)
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out


def _frag_smiles_single_cut_keep_atom(mol: Chem.Mol, a_idx: int, b_idx: int, keep_idx: int) -> str:
    """
    Cut a single bond between a_idx and b_idx and return the fragment SMILES that contains keep_idx.
    The returned fragment contains exactly one dummy '*' (the attachment point).
    """
    m = Chem.Mol(mol)
    for a in m.GetAtoms():
        a.SetAtomMapNum(int(a.GetIdx()) + 1)
    bond = m.GetBondBetweenAtoms(int(a_idx), int(b_idx))
    if bond is None:
        raise ValueError("bond not found for cut")
    frag = Chem.FragmentOnBonds(m, [bond.GetIdx()], addDummies=True, dummyLabels=[(1, 1)])
    frags = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=True)
    keep_map = int(keep_idx) + 1
    for fm in frags:
        maps = {int(a.GetAtomMapNum()) for a in fm.GetAtoms() if int(a.GetAtomMapNum()) > 0}
        if keep_map in maps:
            fm2 = _clear_atom_mapnums(fm)
            smi = Chem.MolToSmiles(fm2, canonical=True, isomericSmiles=False)
            return str(smi)
    raise ValueError("keep atom not found in fragments after cut")


def _assign_R1_R4_from_acetal_smiles_molwt(
    acetal_smiles: str,
    molwt_cache: Dict[str, Dict[str, float]],
    norm_map: Dict[str, str],
) -> Dict[str, str]:
    """
    Assign R1..R4 from a whole-molecule acetal SMILES, strictly following the project's molwt rule
    (README.md section "虚拟分子构建规则（R 位点分配）", i.e. 项目说明-新版.txt).

    Supports cases where the acetal center carbon has 2/3/4 single-bond oxygen neighbors.
    The returned substituent strings use the project's schema (uncapped, with one '*', or '[H]').
    """
    mol = Chem.MolFromSmiles(str(acetal_smiles).strip())
    if mol is None:
        raise ValueError("invalid smiles")
    can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    mol = Chem.MolFromSmiles(can)
    if mol is None:
        raise ValueError("invalid canonical smiles")

    c_idx = _find_acetal_center(mol)
    c_atom = mol.GetAtomWithIdx(int(c_idx))

    # Identify single-bond oxygen neighbors and the remaining neighbors.
    o_nei = []
    other_nei = []
    for n in c_atom.GetNeighbors():
        b = mol.GetBondBetweenAtoms(c_atom.GetIdx(), n.GetIdx())
        if b is None or b.GetBondType() != Chem.rdchem.BondType.SINGLE:
            continue
        if n.GetAtomicNum() == 8:
            o_nei.append(n.GetIdx())
        else:
            other_nei.append(n.GetIdx())

    nO = int(len(o_nei))
    if nO < 2 or nO > 4:
        raise ValueError(f"unexpected number of single-bond O neighbors at acetal center: {nO}")

    def norm_one(x: str) -> str:
        s0 = str(x).strip()
        if s0 in {"", "H"}:
            s0 = "[H]"
        # Canonicalize if possible
        if s0 != "[H]":
            m0 = Chem.MolFromSmiles(s0)
            if m0 is not None:
                s0 = Chem.MolToSmiles(m0, canonical=True, isomericSmiles=False)
        return norm_map.get(s0, s0)

    def mw_of_uncapped(u: str) -> float:
        u = norm_one(u)
        if u in {"[H]"}:
            return 0.0
        rec = molwt_cache.get(u)
        if rec is not None and "molwt_including_attach" in rec:
            return float(rec["molwt_including_attach"])
        cap = _to_capped_smiles_canonical(u)
        m0 = Chem.MolFromSmiles(cap)
        if m0 is None:
            return 0.0
        return float(Descriptors.MolWt(m0))

    # For each oxygen neighbor, extract:
    # - noO token: cut O-R, keep R side (or [H] if no R).
    # - withO token: cut C-O, keep O side (contains O and R).
    o_items = []
    for o_idx in o_nei:
        o_atom = mol.GetAtomWithIdx(int(o_idx))
        r_nei = [x.GetIdx() for x in o_atom.GetNeighbors() if x.GetIdx() != c_idx]
        if not r_nei:
            noO = "[H]"
        else:
            # If multiple neighbors (rare), pick the first deterministically by index.
            r_idx = int(sorted(r_nei)[0])
            noO = _frag_smiles_single_cut_keep_atom(mol, o_idx, r_idx, r_idx)
        withO = _frag_smiles_single_cut_keep_atom(mol, c_idx, o_idx, o_idx)
        noO = norm_one(noO)
        withO = norm_one(withO)
        o_items.append(
            {
                "o_idx": int(o_idx),
                "noO": noO,
                "withO": withO,
                "mw_noO": mw_of_uncapped(noO),
                "mw_withO": mw_of_uncapped(withO),
            }
        )

    # Direct substituents (non-oxygen neighbors): cut C-X, keep X side.
    direct_items = []
    for x_idx in other_nei:
        u = _frag_smiles_single_cut_keep_atom(mol, c_idx, x_idx, x_idx)
        u = norm_one(u)
        direct_items.append({"u": u, "mw": mw_of_uncapped(u)})

    # Pad direct list to length 2 with [H] for convenience.
    while len(direct_items) < 2:
        direct_items.append({"u": "[H]", "mw": 0.0})
    direct_items = direct_items[:2]

    # Apply the molwt rule.
    # Sort oxygen-connected by non-oxygen part MW.
    o_sorted = sorted(
        o_items,
        key=lambda d: (
            float(d["mw_noO"]),
            float(d["mw_withO"]),
            str(d["noO"]),
            str(d["withO"]),
        ),
        reverse=True,
    )

    if nO == 2:
        r1 = o_sorted[0]["noO"]
        r2 = o_sorted[1]["noO"]
        d_sorted = sorted(direct_items, key=lambda d: (float(d["mw"]), str(d["u"])), reverse=True)
        r3 = d_sorted[0]["u"]
        r4 = d_sorted[1]["u"]
        return {"R1_smiles": r1, "R2_smiles": r2, "R3_smiles": r3, "R4_smiles": r4}

    if nO == 3:
        r1 = o_sorted[0]["noO"]
        r2 = o_sorted[1]["noO"]
        # smallest (by mw_noO) goes to R3 with oxygen
        r3 = o_sorted[2]["withO"]
        # remaining direct substituent (should be 1) goes to R4
        r4 = direct_items[0]["u"] if direct_items else "[H]"
        return {"R1_smiles": r1, "R2_smiles": r2, "R3_smiles": r3, "R4_smiles": r4}

    # nO == 4
    r1 = o_sorted[0]["noO"]
    r2 = o_sorted[1]["noO"]
    # remaining two go to R3/R4 with oxygen, in descending mw_withO (tie-breaker mw_noO)
    rest = sorted(
        o_sorted[2:],
        key=lambda d: (float(d["mw_withO"]), float(d["mw_noO"]), str(d["withO"]), str(d["noO"])),
        reverse=True,
    )
    r3 = rest[0]["withO"]
    r4 = rest[1]["withO"]
    return {"R1_smiles": r1, "R2_smiles": r2, "R3_smiles": r3, "R4_smiles": r4}


def _decompose_substituents_project_schema(acetal_smiles: str) -> Tuple[str, List[str], List[str]]:
    """
    Decompose an acetal SMILES into substituents following this project's schema:
      - R1/R2 are the groups on the two "acetal oxygens" (oxygen removed from substituent);
        i.e., we cut O–R bonds (not C_center–O).
      - R3/R4 are the two groups directly attached to the center carbon
        (cut C_center–R bonds).

    Returns:
      canonical_acetal_smiles,
      o_side_subs: 2 substituents (uncapped with '*', or '[H]') corresponding to the two O-side R groups,
      direct_subs: up to 2 substituents corresponding to the direct groups on the center carbon.
    """
    s = str(acetal_smiles).strip()
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        raise ValueError("invalid smiles")
    can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)

    c_idx = _find_acetal_center(mol)
    c_atom = mol.GetAtomWithIdx(int(c_idx))
    o_nei = [n for n in c_atom.GetNeighbors() if n.GetAtomicNum() == 8]
    other_nei = [n for n in c_atom.GetNeighbors() if n.GetAtomicNum() != 8]
    if len(o_nei) < 2:
        raise ValueError("acetal center must have >=2 O neighbors")

    bonds = []
    labels = []
    # Label isotopes: 1/2 for O-side cuts, 3/4 for direct cuts.
    iso = 1
    for o in o_nei[:2]:
        # Cut O–R (R is the neighbor of O other than center carbon)
        r_nei = [x for x in o.GetNeighbors() if x.GetIdx() != c_idx]
        if not r_nei:
            continue
        r = r_nei[0]
        b = mol.GetBondBetweenAtoms(o.GetIdx(), r.GetIdx())
        if b is None:
            continue
        bonds.append(b.GetIdx())
        labels.append((iso, iso))
        iso += 1

    for x in other_nei[:2]:
        b = mol.GetBondBetweenAtoms(c_idx, x.GetIdx())
        if b is None:
            continue
        bonds.append(b.GetIdx())
        labels.append((iso, iso))
        iso += 1

    if len(bonds) == 0:
        raise ValueError("no bonds to cut for substituent decomposition")

    frag = Chem.FragmentOnBonds(mol, bonds, addDummies=True, dummyLabels=labels)
    frags = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=True)

    o_side = []
    direct = []
    for fm in frags:
        d = [a for a in fm.GetAtoms() if a.GetAtomicNum() == 0]
        if len(d) != 1:
            continue
        didx = d[0].GetIdx()
        iso_val = int(fm.GetAtomWithIdx(didx).GetIsotope())
        fm2 = _strip_dummy_isotopes(fm)
        smi = Chem.MolToSmiles(fm2, canonical=True, isomericSmiles=False)
        if iso_val in (1, 2):
            o_side.append(smi)
        elif iso_val in (3, 4):
            direct.append(smi)

    # Pad to expected sizes
    while len(o_side) < 2:
        o_side.append("[H]")
    if len(o_side) > 2:
        o_side = o_side[:2]
    while len(direct) < 2:
        direct.append("[H]")
    if len(direct) > 2:
        direct = direct[:2]

    return can, o_side, direct


def _assign_R1_R4_by_molwt(
    uncapped_subs: List[str],
    molwt_cache: Dict[str, Dict[str, float]],
) -> Dict[str, str]:
    """
    Implement the molwt-based assignment logic described in 项目说明-新版.txt.

    Input subs should be 4 substituents with '*' or '[H]'. We classify them by whether
    the attachment atom is O (neighbor of '*').
    """
    if len(uncapped_subs) != 4:
        raise ValueError("expected 4 substituents")

    # Canonicalize keys for cache lookup (fallback to runtime compute)
    def get_rec(s: str) -> Tuple[str, Dict[str, float]]:
        k = str(s).strip()
        if k in {"H"}:
            k = "[H]"
        if k not in {"[H]"}:
            m = Chem.MolFromSmiles(k)
            if m is not None:
                k = Chem.MolToSmiles(m, canonical=True, isomericSmiles=False)
        # Use cached, else compute quickly and DO NOT persist here (caller may persist).
        rec = molwt_cache.get(k)
        if rec is not None:
            return k, rec
        attach = _attach_atom_symbol(k)
        capped = _to_capped_smiles_canonical(k)
        m = Chem.MolFromSmiles(capped)
        mw = float(Descriptors.MolWt(m)) if m is not None else 0.0
        mw_excl = mw - 15.999 if attach == "O" else mw
        rec2 = {
            "attach_atom_symbol": attach,
            "capped_smiles_canonical": capped,
            "molwt_including_attach": mw,
            "molwt_excluding_attach_O": mw_excl,
        }
        # Update cache in-memory; caller may persist to disk after the run.
        molwt_cache[k] = rec2
        return k, rec2

    # Backward-compatible fallback: if caller does not provide typed groups, we infer by attachment atom.
    recs = [(s, get_rec(s)[1]) for s in uncapped_subs]
    o_subs = [(s, r) for s, r in recs if r.get("attach_atom_symbol") == "O"]
    non_o_subs = [(s, r) for s, r in recs if r.get("attach_atom_symbol") != "O"]

    if len(o_subs) == 2:
        o_sorted = sorted(o_subs, key=lambda t: float(t[1]["molwt_excluding_attach_O"]), reverse=True)
        n_sorted = sorted(non_o_subs, key=lambda t: float(t[1]["molwt_including_attach"]), reverse=True)
        r1 = o_sorted[0][0]
        r2 = o_sorted[1][0]
        r3 = n_sorted[0][0] if len(n_sorted) >= 1 else "[H]"
        r4 = n_sorted[1][0] if len(n_sorted) >= 2 else "[H]"
        return {"R1_smiles": r1, "R2_smiles": r2, "R3_smiles": r3, "R4_smiles": r4}

    # If we can't infer by attach atom, fail and let caller provide correct typed decomposition.
    raise ValueError("cannot assign R1-R4 reliably from untyped substituent list; use project-schema decomposition")


def _assign_R1_R4_by_molwt_typed(
    o_side_subs: List[str],
    direct_subs: List[str],
    molwt_cache: Dict[str, Dict[str, float]],
) -> Dict[str, str]:
    """
    Assign R1..R4 by the project's molwt rule for the common case (README.md section "R 位点分配", situation B):

      Acetal: R1-O-C(R3)(R4)-O-R2

    Inputs:
      - o_side_subs: 2 substituents attached to the two acetal oxygens, with '*' on the attachment atom
        and without the connecting oxygen (e.g. "CC*" rather than "*OCC"). Missing -> "[H]".
      - direct_subs: up to 2 substituents directly attached to the center carbon, with '*'.
        Missing -> "[H]".

    Rule:
      - R1/R2: sort the two O-side substituents by molwt (of methyl-capped substituent) descending.
      - R3/R4: sort the two direct substituents by molwt (of methyl-capped substituent) descending.

    Note:
      We use the cached "molwt_including_attach" computed from the methyl-capped SMILES, which matches
      the "不含氧部分分子量" requirement because the connecting oxygen is already removed in this schema.
    """
    if len(o_side_subs) != 2:
        raise ValueError("expected 2 O-side substituents")
    if len(direct_subs) != 2:
        raise ValueError("expected 2 direct substituents")

    def get_rec(s: str) -> Tuple[str, Dict[str, float]]:
        k = str(s).strip()
        if k in {"H"}:
            k = "[H]"
        if k != "[H]":
            m = Chem.MolFromSmiles(k)
            if m is not None:
                k = Chem.MolToSmiles(m, canonical=True, isomericSmiles=False)

        rec = molwt_cache.get(k)
        if rec is not None:
            return k, rec

        # Compute-on-miss and keep in cache (persisted by main()).
        attach = _attach_atom_symbol(k)
        capped = _to_capped_smiles_canonical(k)
        m = Chem.MolFromSmiles(capped)
        mw = float(Descriptors.MolWt(m)) if m is not None else 0.0
        # Legacy field: for older schemas with "*O..." keys, exclude the connecting oxygen.
        mw_excl = mw - 15.999 if attach == "O" else mw
        rec2 = {
            "attach_atom_symbol": str(attach),
            "capped_smiles_canonical": str(capped),
            "molwt_including_attach": float(mw),
            "molwt_excluding_attach_O": float(mw_excl),
        }
        molwt_cache[k] = rec2
        return k, rec2

    o = [get_rec(s) for s in o_side_subs]
    d = [get_rec(s) for s in direct_subs]

    o_sorted = sorted(o, key=lambda t: float(t[1].get("molwt_including_attach", 0.0)), reverse=True)
    d_sorted = sorted(d, key=lambda t: float(t[1].get("molwt_including_attach", 0.0)), reverse=True)

    return {
        "R1_smiles": o_sorted[0][0],
        "R2_smiles": o_sorted[1][0],
        "R3_smiles": d_sorted[0][0],
        "R4_smiles": d_sorted[1][0],
    }


def _load_molwt_cache_csv(path: Path) -> Dict[str, Dict[str, float]]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    need = {
        "substituent_uncapped_canon",
        "attach_atom_symbol",
        "capped_smiles_canonical",
        "molwt_including_attach",
        "molwt_excluding_attach_O",
    }
    if not need.issubset(set(df.columns)):
        raise ValueError(f"molwt cache missing columns: {sorted(need - set(df.columns))}")
    m = {}
    for _, r in df.iterrows():
        k = str(r["substituent_uncapped_canon"]).strip()
        m[k] = {
            "attach_atom_symbol": str(r["attach_atom_symbol"]).strip(),
            "capped_smiles_canonical": str(r["capped_smiles_canonical"]).strip(),
            "molwt_including_attach": float(r["molwt_including_attach"]),
            "molwt_excluding_attach_O": float(r["molwt_excluding_attach_O"]),
        }
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack_dir", required=True, type=str)
    ap.add_argument("--input_csv", required=True, type=str, help="CSV with at least: smiles (and optionally molecule)")
    ap.add_argument("--out_csv", required=True, type=str)
    ap.add_argument("--molecule_col", default="molecule", type=str)
    ap.add_argument("--smiles_col", default="smiles", type=str)
    ap.add_argument(
        "--use_positions_if_present",
        action="store_true",
        help="If input_csv includes R1_smiles..R4_smiles, use them directly (exactly reproducible with step4 outputs).",
    )
    ap.add_argument("--disable_feasible", action="store_true", help="Disable SA/logP hard constraints (for verification).")
    ap.add_argument("--disable_novel", action="store_true", help="Disable novelty requirement (for verification).")
    ap.add_argument("--chunksize", default=5000, type=int)
    ap.add_argument("--batch_size", default=4096, type=int)
    ap.add_argument("--device", default="cpu", type=str)
    ap.add_argument("--write_positions", action="store_true", help="Write assigned R1..R4 in output.")
    args = ap.parse_args()

    require_feasible = not bool(args.disable_feasible)
    require_novel = not bool(args.disable_novel)

    pack = Path(args.pack_dir)
    model_dir = pack / "models" / "logk2_dnn3_seed3"
    ensemble_dir = pack / "models" / "logk2_dnn3_5seed_ensemble"
    schema_csv = pack / "schemas" / "x51d" / "feature_mapping_51d.csv"
    template = pack / "schemas" / "x51d" / "virt_molwt_v1_full_X51D_from1500eprops_cdftFixed.csv.gz"
    # Prefer merged lookup (base + newly DFT-computed substituents) if present.
    cdft_csv = pack / "props" / "cdft9_lookup" / "cdft6_lookup_merged.csv"
    if not cdft_csv.exists():
        cdft_csv = pack / "props" / "cdft9_lookup" / "cdft_multiwfn_merged-substituent_dedup.csv"
    known_csv_gz = pack / "data" / "known_molecules" / "known_canonical_smiles_71w184.csv.gz"
    molwt_cache_csv = pack / "data" / "substituents" / "base_new" / "substituent_molwt_cache.csv"
    norm_map_csv = pack / "data" / "substituents" / "base_new" / "substituent_normalization_map.csv"
    suite_root_txt = pack / "models" / "eprops8_suite" / "ORIGINAL_SUITE_ROOT.txt"

    for p in [model_dir, ensemble_dir, schema_csv, template, cdft_csv, known_csv_gz, molwt_cache_csv, suite_root_txt]:
        if not Path(p).exists():
            raise FileNotFoundError(str(p))

    gids = _read_gids_from_template(template)
    schema = _load_51d_schema(schema_csv)
    specs = [schema[g] for g in gids]

    needed_cdft_props = sorted({s.prop for s in specs if s.block == "sub" and s.prop})
    cdft_lookup = _load_cdft_lookup(cdft_csv, needed_cdft_props)
    needed_eprops = [s.label for s in specs if s.block == "eprops"]

    known = _load_known_canonical_smiles_set(known_csv_gz) if require_novel else set()
    molwt_cache = _load_molwt_cache_csv(molwt_cache_csv)
    molwt_cache_n0 = len(molwt_cache)
    norm_map = _load_norm_map(norm_map_csv)

    sascorer = _load_sascorer()
    # load 5-seed ensemble members
    member_dirs = sorted([p for p in Path(ensemble_dir).glob("member_*") if p.is_dir()])
    pred = DNN3EnsemblePredictor(member_dirs=member_dirs, device=str(args.device))
    suite_root = Path(suite_root_txt.read_text(encoding="utf-8").replace("\ufeff", "").strip())
    if not suite_root.exists():
        raise FileNotFoundError(f"eprops suite root from {suite_root_txt} not found: {suite_root}")
    eprops = EpropsSuitePredictor(suite_root=suite_root)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    # Caches
    sub_desc_cache: Dict[str, Dict[str, float]] = {}  # capped_smiles -> desc dict
    cdft_cache: Dict[str, Dict[str, float]] = {}  # capped_smiles -> cdft dict
    global_cache: Dict[str, Dict[str, float]] = {}  # canonical acetal smiles -> global10 dict
    sa_logp_cache: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    eprops_cache: Dict[str, Dict[str, float]] = {}  # canonical acetal smiles -> eprops dict

    wrote = False

    for chunk in pd.read_csv(Path(args.input_csv), chunksize=int(args.chunksize), dtype=str, encoding="utf-8-sig"):
        if args.smiles_col not in chunk.columns:
            raise ValueError(f"input_csv missing smiles_col={args.smiles_col}. cols={list(chunk.columns)[:30]}")
        mol_col = args.molecule_col if args.molecule_col in chunk.columns else None
        smiles_list = chunk[args.smiles_col].astype(str).str.strip().tolist()
        mol_ids = chunk[mol_col].astype(str).tolist() if mol_col else ["" for _ in smiles_list]

        r_cols = ["R1_smiles", "R2_smiles", "R3_smiles", "R4_smiles"]
        has_positions = bool(args.use_positions_if_present) and all(c in chunk.columns for c in r_cols)
        if has_positions:
            r1_list = chunk["R1_smiles"].astype(str).str.strip().tolist()
            r2_list = chunk["R2_smiles"].astype(str).str.strip().tolist()
            r3_list = chunk["R3_smiles"].astype(str).str.strip().tolist()
            r4_list = chunk["R4_smiles"].astype(str).str.strip().tolist()

        # Pass 1: compute SA/logP, canonical, novelty, feasibility
        can_list = []
        feas_mask = []
        sa_list = []
        lp_list = []
        novel_list = []
        o_side_list: List[List[str]] = []
        direct_list: List[List[str]] = []
        pos_list: List[Dict[str, str]] = []

        for smi in smiles_list:
            can = _canon_mol_smiles(smi)
            can_list.append(can or "")
            if not can:
                sa_list.append(np.nan)
                lp_list.append(np.nan)
                novel_list.append(False)
                feas_mask.append(False)
                o_side_list.append(["[H]", "[H]"])
                direct_list.append(["[H]", "[H]"])
                pos_list.append({"R1_smiles": "[H]", "R2_smiles": "[H]", "R3_smiles": "[H]", "R4_smiles": "[H]"})
                continue
            if can in sa_logp_cache:
                sa, lp = sa_logp_cache[can]
            else:
                sa, lp = _calc_sa_logp(can, sascorer=sascorer)
                sa_logp_cache[can] = (sa, lp)
            sa_list.append(sa if sa is not None else np.nan)
            lp_list.append(lp if lp is not None else np.nan)

            is_feas = bool(sa is not None and lp is not None and sa < 5 and lp > -1 and lp < 5) if require_feasible else True
            is_novel = bool(can not in known) if require_novel else True
            novel_list.append(is_novel)
            feas_mask.append(bool(is_feas and is_novel))

            def _norm_one(x: str) -> str:
                s0 = str(x).strip()
                if s0 in {"H"}:
                    s0 = "[H]"
                return norm_map.get(s0, s0)

            if has_positions:
                idx = len(can_list) - 1
                pos_list.append(
                    {
                        "R1_smiles": _norm_one(r1_list[idx]),
                        "R2_smiles": _norm_one(r2_list[idx]),
                        "R3_smiles": _norm_one(r3_list[idx]),
                        "R4_smiles": _norm_one(r4_list[idx]),
                    }
                )
                # Keep placeholders (won't be used when has_positions=True).
                o_side_list.append(["[H]", "[H]"])
                direct_list.append(["[H]", "[H]"])
            else:
                try:
                    pos = _assign_R1_R4_from_acetal_smiles_molwt(can, molwt_cache=molwt_cache, norm_map=norm_map)
                except Exception:
                    pos = {"R1_smiles": "[H]", "R2_smiles": "[H]", "R3_smiles": "[H]", "R4_smiles": "[H]"}
                pos_list.append(pos)
                o_side_list.append(["[H]", "[H]"])
                direct_list.append(["[H]", "[H]"])

        # Prepare eprops predictions only for feasible candidates
        idx_keep = [i for i, ok in enumerate(feas_mask) if ok]
        if idx_keep:
            smiles_keep = [can_list[i] for i in idx_keep]
            missing_ep = [s for s in smiles_keep if s not in eprops_cache]
            if missing_ep:
                ep_df = eprops.predict(missing_ep, batch_size=int(args.batch_size))
                for _, r in ep_df.iterrows():
                    cs = str(r["smiles"]).strip()
                    rec = {t: float(r[t]) for t in needed_eprops}
                    eprops_cache[cs] = rec

        # Build 51D features and predict for keepers
        rows_out = []
        if idx_keep:
            vecs: List[np.ndarray] = []
            for j, i in enumerate(idx_keep):
                can = can_list[i]
                if has_positions:
                    pos = pos_list[i]
                else:
                    pos = pos_list[i]
                    if not pos or not all(k in pos for k in ["R1_smiles", "R2_smiles", "R3_smiles", "R4_smiles"]):
                        continue

                # Compute per-position substituent descriptors + cdft (lookup by capped canonical)
                sub_desc_by_pos = {}
                cdft_by_pos = {}
                for p in ["R1", "R2", "R3", "R4"]:
                    unc = pos[f"{p}_smiles"]
                    capped = _to_capped_smiles_canonical(unc)
                    if capped in sub_desc_cache:
                        sub_desc = sub_desc_cache[capped]
                    else:
                        m = Chem.MolFromSmiles(capped)
                        sub_desc = _compute_sub_rdkit_descs(m) if m is not None else {k: 0.0 for k in _compute_sub_rdkit_descs(Chem.MolFromSmiles("C"))}
                        sub_desc_cache[capped] = sub_desc
                    sub_desc_by_pos[p] = sub_desc

                    if capped in cdft_cache:
                        cd = cdft_cache[capped]
                    else:
                        cd = cdft_lookup.get(capped)
                        if cd is None:
                            # missing cdft: cannot evaluate reliably in this setup -> discard
                            cd = {}
                        cdft_cache[capped] = cd
                    cdft_by_pos[p] = cd

                # Global10 desc
                if can in global_cache:
                    gdesc = global_cache[can]
                else:
                    m = Chem.MolFromSmiles(can)
                    gdesc = _compute_global10_desc(m) if m is not None else {k: 0.0 for k in _compute_global10_desc(Chem.MolFromSmiles("C"))}
                    global_cache[can] = gdesc

                ep = eprops_cache.get(can, {})

                # Fill X in gid order
                vec = []
                ok = True
                for s in specs:
                    if s.block == "rdkit":
                        v = float(sub_desc_by_pos[s.position][s.prop])
                    elif s.block == "global":
                        v = float(gdesc[s.label])
                    elif s.block == "sub":
                        cd = cdft_by_pos[s.position]
                        if s.prop not in cd:
                            ok = False
                            v = 0.0
                        else:
                            v = float(cd[s.prop])
                    elif s.block == "eprops":
                        if s.label not in ep:
                            ok = False
                            v = 0.0
                        else:
                            v = float(ep[s.label])
                    else:
                        ok = False
                        v = 0.0
                    vec.append(v)
                if not ok:
                    continue
                vecs.append(np.asarray(vec, dtype=np.float32))

                rows_out.append(
                    {
                        "molecule": mol_ids[i],
                        "smiles": can,
                        "sa_score": sa_list[i],
                        "logP": lp_list[i],
                        "novel": True,
                        "is_feasible": True,
                        "R1_smiles": pos["R1_smiles"] if args.write_positions else None,
                        "R2_smiles": pos["R2_smiles"] if args.write_positions else None,
                        "R3_smiles": pos["R3_smiles"] if args.write_positions else None,
                        "R4_smiles": pos["R4_smiles"] if args.write_positions else None,
                    }
                )

            if rows_out:
                Xmat = np.vstack(vecs).astype(np.float32, copy=False)
                yhat_mean, yhat_std = pred.predict_mean_std(X_raw_51=Xmat, batch_size=int(args.batch_size))
                for k in range(len(rows_out)):
                    rows_out[k]["pred_LOGk2_mean_seeds"] = float(yhat_mean[k])
                    rows_out[k]["pred_LOGk2_std_seeds"] = float(yhat_std[k])

        # Add non-feasible rows (optional for auditing)
        for i in range(len(smiles_list)):
            if feas_mask[i]:
                continue
            rows_out.append(
                {
                    "molecule": mol_ids[i],
                    "smiles": can_list[i] or smiles_list[i],
                    "sa_score": sa_list[i],
                    "logP": lp_list[i],
                    "novel": bool(novel_list[i]),
                    "is_feasible": False,
                    "pred_LOGk2_mean_seeds": np.nan,
                    "pred_LOGk2_std_seeds": np.nan,
                    "R1_smiles": None,
                    "R2_smiles": None,
                    "R3_smiles": None,
                    "R4_smiles": None,
                }
            )

        out_df = pd.DataFrame(rows_out)
        # Drop position cols if not requested
        if not args.write_positions:
            out_df = out_df.drop(columns=["R1_smiles", "R2_smiles", "R3_smiles", "R4_smiles"], errors="ignore")
        out_df.to_csv(out_path, mode="a", index=False, header=(not wrote), encoding="utf-8-sig")
        wrote = True

    print("[OK] wrote:", out_path, flush=True)

    # Persist molwt cache if new substituents were encountered (future-proofing).
    if len(molwt_cache) != molwt_cache_n0:
        rows = []
        for k, r in molwt_cache.items():
            rows.append(
                {
                    "substituent_uncapped_canon": k,
                    "attach_atom_symbol": r.get("attach_atom_symbol", ""),
                    "capped_smiles_canonical": r.get("capped_smiles_canonical", ""),
                    "molwt_including_attach": float(r.get("molwt_including_attach", 0.0)),
                    "molwt_excluding_attach_O": float(r.get("molwt_excluding_attach_O", 0.0)),
                }
            )
        pd.DataFrame(rows).sort_values(["substituent_uncapped_canon"]).to_csv(
            molwt_cache_csv, index=False, encoding="utf-8-sig"
        )
        print(f"[OK] updated molwt cache: {molwt_cache_n0} -> {len(molwt_cache)} at {molwt_cache_csv}", flush=True)


if __name__ == "__main__":
    main()
