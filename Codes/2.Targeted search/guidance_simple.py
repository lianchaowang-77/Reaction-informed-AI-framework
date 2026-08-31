from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rdkit import Chem


POS = ("R1", "R2", "R3", "R4")


def _canon_sub(s: str) -> Optional[str]:
    s0 = str(s).strip()
    if not s0:
        return None
    if s0 == "H":
        s0 = "[H]"
    m = Chem.MolFromSmiles(s0)
    if m is None:
        return None
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=False)


def load_guidance(combo_dir: Path, pools: Dict[str, List[str]]) -> Dict[str, List[dict]]:
    """
    Load high-frequency pair/triple combinations directly (no variant expansion).
    Only retain combos fully present in current pools.
    """
    combo_dir = Path(combo_dir)
    pair_csv = combo_dir / "high4_pair_combinations_top50_each.csv"
    tri_csv = combo_dir / "high4_triple_combinations_top50_each.csv"

    pool_sets = {p: set(pools[p]) for p in POS}
    out = {"pairs": [], "triples": []}

    if pair_csv.exists():
        df = pd.read_csv(pair_csv, encoding="utf-8-sig")
        for _, r in df.iterrows():
            p1, p2 = str(r.get("pos1", "")).strip(), str(r.get("pos2", "")).strip()
            s1, s2 = _canon_sub(str(r.get("sub1", "")).strip()), _canon_sub(str(r.get("sub2", "")).strip())
            if p1 not in POS or p2 not in POS or p1 == p2:
                continue
            if not s1 or not s2:
                continue
            if s1 not in pool_sets[p1] or s2 not in pool_sets[p2]:
                continue
            w = float(r.get("search_weight", r.get("freq", r.get("count", 1.0))))
            guide_id = str(r.get("guide_id", f"{p1}-{p2}:{s1}|{s2}")).strip()
            out["pairs"].append(
                {
                    "assign": {p1: s1, p2: s2},
                    "w": max(w, 1e-12),
                    "guide_id": guide_id,
                    "guide_type": "pair",
                }
            )

    if tri_csv.exists():
        df = pd.read_csv(tri_csv, encoding="utf-8-sig")
        for _, r in df.iterrows():
            p1, p2, p3 = str(r.get("pos1", "")).strip(), str(r.get("pos2", "")).strip(), str(r.get("pos3", "")).strip()
            s1 = _canon_sub(str(r.get("sub1", "")).strip())
            s2 = _canon_sub(str(r.get("sub2", "")).strip())
            s3 = _canon_sub(str(r.get("sub3", "")).strip())
            if p1 not in POS or p2 not in POS or p3 not in POS:
                continue
            if len({p1, p2, p3}) != 3:
                continue
            if not s1 or not s2 or not s3:
                continue
            if s1 not in pool_sets[p1] or s2 not in pool_sets[p2] or s3 not in pool_sets[p3]:
                continue
            w = float(r.get("search_weight", r.get("freq", r.get("count", 1.0))))
            guide_id = str(r.get("guide_id", f"{p1}-{p2}-{p3}:{s1}|{s2}|{s3}")).strip()
            out["triples"].append(
                {
                    "assign": {p1: s1, p2: s2, p3: s3},
                    "w": max(w, 1e-12),
                    "guide_id": guide_id,
                    "guide_type": "triple",
                }
            )

    return out


def _weighted_pick(items: List[dict], rng: random.Random) -> dict:
    ws = [float(max(x.get("w", 1.0), 1e-12)) for x in items]
    s = sum(ws)
    t = rng.random() * s
    acc = 0.0
    for it, w in zip(items, ws):
        acc += w
        if acc >= t:
            return it
    return items[-1]


def sample_guided_assignment(
    pools: Dict[str, List[str]],
    guidance: Optional[Dict[str, List[dict]]],
    rng: random.Random,
    triple_prefer: float = 0.6,
) -> Dict[str, str]:
    assignment, _ = sample_guided_assignment_with_meta(
        pools,
        guidance,
        rng=rng,
        triple_prefer=triple_prefer,
    )
    return assignment


def sample_guided_assignment_with_meta(
    pools: Dict[str, List[str]],
    guidance: Optional[Dict[str, List[dict]]],
    rng: random.Random,
    triple_prefer: float = 0.6,
) -> Tuple[Dict[str, str], Dict[str, object]]:
    # start random
    a = {p: rng.choice(pools[p]) for p in POS}
    if not guidance:
        return a, {"is_guided": False, "guide_id": "", "guide_type": "", "guide_weight": 0.0}

    triples = guidance.get("triples", [])
    pairs = guidance.get("pairs", [])

    if triples and (not pairs or rng.random() < float(triple_prefer)):
        pick = _weighted_pick(triples, rng)
        a.update(pick["assign"])
        return a, {
            "is_guided": True,
            "guide_id": str(pick.get("guide_id", "")),
            "guide_type": str(pick.get("guide_type", "triple")),
            "guide_weight": float(pick.get("w", 0.0)),
        }
    if pairs:
        pick = _weighted_pick(pairs, rng)
        a.update(pick["assign"])
        return a, {
            "is_guided": True,
            "guide_id": str(pick.get("guide_id", "")),
            "guide_type": str(pick.get("guide_type", "pair")),
            "guide_weight": float(pick.get("w", 0.0)),
        }
    return a, {"is_guided": False, "guide_id": "", "guide_type": "", "guide_weight": 0.0}
