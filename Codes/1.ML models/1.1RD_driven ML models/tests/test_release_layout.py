from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "DNN": "DNN3.pt",
    "Ridge": "model.pkl",
    "Lasso": "model.pkl",
    "SVR": "model.pkl",
    "LGBM": "model.pkl",
    "XGB": "model.pkl",
    "RF": "model.pkl",
    "ET": "model.pkl",
}


class ReleaseLayoutTests(unittest.TestCase):
    def test_release_contains_eight_independent_model_entry_points(self) -> None:
        self.assertTrue((ROOT / "rdml" / "__init__.py").is_file())
        for model_name in MODELS:
            model_dir = ROOT / model_name
            self.assertTrue(model_dir.is_dir(), model_name)
            self.assertTrue((model_dir / "train.py").is_file(), model_name)
            self.assertTrue((model_dir / "predict.py").is_file(), model_name)
            self.assertTrue((model_dir / "model_config.json").is_file(), model_name)

    def test_each_model_has_five_complete_pretrained_members(self) -> None:
        for model_name, weight_name in MODELS.items():
            pretrained = ROOT / model_name / "pretrained"
            self.assertTrue(pretrained.is_dir(), model_name)
            self.assertEqual(
                sorted(p.name for p in pretrained.iterdir() if p.is_dir()),
                ["seed1", "seed2", "seed3", "seed4", "seed5"],
            )
            for release_seed in range(1, 6):
                member = pretrained / f"seed{release_seed}"
                self.assertTrue((member / weight_name).is_file())
                self.assertTrue((member / "Xdesc_scaler.pkl").is_file())
                self.assertTrue((member / "Yscaler.pkl").is_file())
                self.assertTrue((member / "config.json").is_file())

    def test_release_excludes_result_and_analysis_files(self) -> None:
        forbidden_suffixes = {".csv", ".log", ".png", ".jpg", ".jpeg", ".svg", ".pdf", ".sqlite", ".bak", ".npy", ".pyc"}
        forbidden_tokens = {
            "summary_predictions",
            "benchmark_baselines",
            "shap",
            "screen",
            "plot",
            "figure",
            "metrics",
            "analysis",
        }
        offending = []
        for path in ROOT.rglob("*"):
            if not path.is_file():
                continue
            name = path.name.lower()
            if path.suffix.lower() in forbidden_suffixes or any(token in name for token in forbidden_tokens):
                offending.append(path.relative_to(ROOT).as_posix())
        offending.extend(
            path.relative_to(ROOT).as_posix()
            for path in ROOT.rglob("__pycache__")
            if path.is_dir()
        )
        self.assertEqual(offending, [])

    def test_dnn_release_configuration_contains_only_sequential_public_ids(self) -> None:
        dnn_root = ROOT / "DNN"
        for release_seed in range(1, 6):
            path = dnn_root / "pretrained" / f"seed{release_seed}" / "config.json"
            self.assertTrue(path.is_file(), path)
            config = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(config["seed"], release_seed)
            self.assertNotIn("member", config)
            self.assertNotIn("source", config)
            self.assertNotIn("original_seed", config)
            self.assertNotIn("source_seed", config)
            serialized = json.dumps(config, ensure_ascii=False)
            self.assertNotIn("member_", serialized)
            self.assertNotIn("benchmark_multi-seed", serialized)
            self.assertNotIn(":\\", serialized)

    def test_release_contains_no_machine_specific_absolute_paths(self) -> None:
        offending = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".py", ".json", ".md", ".txt"}:
                continue
            text = path.read_text(encoding="utf-8")
            if re.search(r"[A-Za-z]:\\", text):
                offending.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(offending, [])


if __name__ == "__main__":
    unittest.main()
