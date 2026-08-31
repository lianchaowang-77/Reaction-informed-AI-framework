from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


PYTHON = Path(r"E:\software\miniconda\anzhuang\envs\lumia\python.exe")
ROOT = Path(__file__).resolve().parent
PACK = Path(r"E:\2025\ML\permeability\PolymerGasMembraneML\datasets\acetal\pack_uncertainty_5seed")
COMBO = ROOT / "guidance_inputs"

MAX_RUNS = 50
MIN_RUNS_BEFORE_RUN_CONVERGENCE = 3
RUN_CONV_PATIENCE = 3
RUN_CONV_DUP_THRESHOLD = 0.95

EVALS_PER_RUN = 20000
BASE_SEED = 1
HIT_THRESHOLD = 6.1
GUIDED_RATIO = 0.30
EFFECTIVE_BATCH = 200
ITER_NOHIT_PATIENCE = 3
ITER_NOHIT_MIN_EVALS = 5000
OUTPUT_ROOT = ROOT / "results"


METHODS = [
    ("BO", "bo.py", "BO"),
    ("GA", "ga.py", "GA"),
    ("NSGA2", "nsga2.py", "NSGA2"),
    ("RL", "rl.py", "RL"),
    ("MCTS", "mcts.py", "MCTS"),
]

METHOD_BATCH_ARGS = {
    "BO": ["--n_init", str(EFFECTIVE_BATCH), "--propose_batch", str(EFFECTIVE_BATCH)],
    "GA": ["--pop_size", str(EFFECTIVE_BATCH)],
    "NSGA2": ["--pop_size", str(EFFECTIVE_BATCH)],
    "RL": ["--sample_batch", str(EFFECTIVE_BATCH)],
    "MCTS": ["--sim_batch", str(EFFECTIVE_BATCH)],
}


def _run_seed(global_run: int) -> int:
    # Method scripts use seed + r * 10007 internally. Because each subprocess
    # runs with --runs 1, set seed so its internal run_01 seed matches the
    # original global run seed BASE_SEED + global_run * 10007.
    return int(BASE_SEED + (int(global_run) - 1) * 10007)


def _method_command(method: str, script: Path, out_dir: Path, run_idx: int) -> list[str]:
    command = [
        str(PYTHON),
        str(script),
        "--pack_dir",
        str(PACK),
        "--out_dir",
        str(out_dir),
        "--runs",
        "1",
        "--evals_per_run",
        str(EVALS_PER_RUN),
        "--seed",
        str(_run_seed(run_idx)),
        "--device",
        "cpu",
        "--batch_size",
        "4096",
        "--combo_dir",
        str(COMBO),
        "--guided_ratio",
        str(GUIDED_RATIO),
        "--iter_nohit_stop",
        "--iter_nohit_threshold",
        str(HIT_THRESHOLD),
        "--iter_nohit_patience",
        str(ITER_NOHIT_PATIENCE),
        "--iter_nohit_min_evals",
        str(ITER_NOHIT_MIN_EVALS),
    ]
    command.extend(METHOD_BATCH_ARGS[method])
    return command


def _read_hits(eval_csv: Path) -> set[str]:
    if not eval_csv.exists():
        return set()
    df = pd.read_csv(eval_csv)
    if "smiles" not in df.columns or "pred_LOGk2_mean_seeds" not in df.columns:
        return set()
    pred = pd.to_numeric(df["pred_LOGk2_mean_seeds"], errors="coerce")
    return set(df.loc[pred > float(HIT_THRESHOLD), "smiles"].astype(str).dropna().tolist())


def _safe_move_run(tmp_run_dir: Path, final_run_dir: Path) -> None:
    if final_run_dir.exists():
        shutil.rmtree(final_run_dir)
    shutil.move(str(tmp_run_dir), str(final_run_dir))


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def run_method(method: str, script_name: str, out_name: str) -> None:
    script = ROOT / script_name
    final_dir = OUTPUT_ROOT / out_name
    final_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = final_dir / "_tmp_single_run"

    cumulative_hits: set[str] = set()
    consecutive_high_dup = 0
    run_records: list[dict] = []
    started = time.time()

    for run_idx in range(1, int(MAX_RUNS) + 1):
        final_run_dir = final_dir / f"run_{run_idx:02d}"
        if final_run_dir.exists() and (final_run_dir / "evaluated.csv").exists():
            print(f"[SKIP-RUN] {method} run_{run_idx:02d} already exists; including in convergence stats.", flush=True)
        else:
            if tmp_root.exists():
                shutil.rmtree(tmp_root)
            tmp_root.mkdir(parents=True, exist_ok=True)
            cmd = _method_command(method, script, tmp_root, run_idx)
            print(f"[RUN] {method} run_{run_idx:02d}: {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, check=True)
            tmp_run_dir = tmp_root / "run_01"
            if not tmp_run_dir.exists():
                raise RuntimeError(f"{method} run_{run_idx:02d} finished but {tmp_run_dir} was not created")
            _safe_move_run(tmp_run_dir, final_run_dir)

        eval_csv = final_run_dir / "evaluated.csv"
        meta = _read_json(final_run_dir / "meta.json")
        hits = _read_hits(eval_csv)
        n_hits = len(hits)
        overlap = len(hits & cumulative_hits)
        new_hits = len(hits - cumulative_hits)
        dup_rate = (overlap / n_hits) if n_hits > 0 else None
        cumulative_before = len(cumulative_hits)
        cumulative_hits |= hits
        cumulative_after = len(cumulative_hits)

        check_enabled = run_idx > int(MIN_RUNS_BEFORE_RUN_CONVERGENCE)
        high_dup = bool(check_enabled and n_hits > 0 and dup_rate is not None and dup_rate >= float(RUN_CONV_DUP_THRESHOLD))
        consecutive_high_dup = consecutive_high_dup + 1 if high_dup else 0
        run_converged = bool(check_enabled and consecutive_high_dup >= int(RUN_CONV_PATIENCE))

        record = {
            "method": method,
            "run": int(run_idx),
            "oracle_evals": meta.get("oracle_evals", meta.get("actual_oracle_evals", "")),
            "iter_nohit_early_stopped": meta.get("iter_nohit_early_stopped", ""),
            "n_gt6p1_unique_in_run": int(n_hits),
            "overlap_with_previous": int(overlap),
            "new_gt6p1_unique": int(new_hits),
            "duplicate_rate_vs_previous": float(dup_rate) if dup_rate is not None else "",
            "cumulative_before": int(cumulative_before),
            "cumulative_after": int(cumulative_after),
            "run_convergence_check_enabled": bool(check_enabled),
            "high_duplicate_run": bool(high_dup),
            "consecutive_high_duplicate_runs": int(consecutive_high_dup),
            "run_converged_after_this_run": bool(run_converged),
        }
        run_records.append(record)
        pd.DataFrame(run_records).to_csv(final_dir / "run_convergence_stats.csv", index=False, encoding="utf-8-sig")

        print(
            f"[STATS] {method} run_{run_idx:02d}: hits={n_hits}, new={new_hits}, "
            f"dup_rate={dup_rate if dup_rate is not None else 'NA'}, "
            f"consecutive_high_dup={consecutive_high_dup}, cumulative_hits={cumulative_after}",
            flush=True,
        )

        if run_converged:
            print(
                f"[RUN-CONVERGED] {method}: stopped after run_{run_idx:02d}; "
                f"{RUN_CONV_PATIENCE} consecutive runs exceeded duplicate rate {RUN_CONV_DUP_THRESHOLD:.2%}.",
                flush=True,
            )
            break

    if tmp_root.exists():
        shutil.rmtree(tmp_root)

    # Build method-level summary from completed run records.
    run_df = pd.DataFrame(run_records)
    summary = {
        "method": method,
        "max_runs": int(MAX_RUNS),
        "completed_runs": int(len(run_records)),
        "evals_per_run_upper_bound": int(EVALS_PER_RUN),
        "effective_evals_per_iteration": int(EFFECTIVE_BATCH),
        "guided_ratio": float(GUIDED_RATIO),
        "iter_nohit_patience": int(ITER_NOHIT_PATIENCE),
        "iter_nohit_min_evals": int(ITER_NOHIT_MIN_EVALS),
        "hit_threshold": float(HIT_THRESHOLD),
        "run_duplicate_threshold": float(RUN_CONV_DUP_THRESHOLD),
        "run_convergence_patience": int(RUN_CONV_PATIENCE),
        "run_convergence_started_after_run": int(MIN_RUNS_BEFORE_RUN_CONVERGENCE),
        "total_unique_gt6p1": int(len(cumulative_hits)),
        "run_converged": bool(run_records[-1]["run_converged_after_this_run"]) if run_records else False,
        "elapsed_sec": float(time.time() - started),
    }
    pd.DataFrame([summary]).to_csv(final_dir / "method_run_convergence_summary.csv", index=False, encoding="utf-8-sig")
    (final_dir / "method_run_convergence_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    if not PYTHON.exists():
        raise FileNotFoundError(str(PYTHON))
    all_summaries = []
    for method, script_name, out_name in METHODS:
        run_method(method, script_name, out_name)
        summary_csv = OUTPUT_ROOT / out_name / "method_run_convergence_summary.csv"
        if summary_csv.exists():
            all_summaries.append(pd.read_csv(summary_csv))
    if all_summaries:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        pd.concat(all_summaries, ignore_index=True).to_csv(OUTPUT_ROOT / "run_convergence_5methods_summary.csv", index=False, encoding="utf-8-sig")
    print("[DONE] all methods finished or run-converged.", flush=True)


if __name__ == "__main__":
    main()
