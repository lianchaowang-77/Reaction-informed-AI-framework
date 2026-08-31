from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGETS = (
    "Cacetal-q(N)",
    "Cacetal-q(N+1)",
    "Cacetal-s-",
    "E_HOMO(N-1)",
    "VIP",
    "Overall_Average",
    "Pos_Average",
    "Polar_Area",
)


class ReleaseLayoutTests(unittest.TestCase):
    def test_release_has_eight_target_entry_points(self) -> None:
        for target in TARGETS:
            directory = ROOT / target
            with self.subTest(target=target):
                self.assertTrue((directory / "train.py").is_file())
                self.assertTrue((directory / "predict.py").is_file())
                self.assertTrue((directory / "model_config.json").is_file())

    def test_each_target_has_complete_pretrained_artifacts(self) -> None:
        for target in TARGETS:
            pretrained = ROOT / target / "pretrained"
            with self.subTest(target=target):
                self.assertTrue((pretrained / "electronic_model.keras").is_file())
                self.assertTrue((pretrained / "Xdesc_scaler.pkl").is_file())
                self.assertTrue((pretrained / "Y_scaler.pkl").is_file())
                self.assertTrue((pretrained / "config.json").is_file())

    def test_suite_entry_points_and_metadata_exist(self) -> None:
        for relative in (
            "train_all.py",
            "predict_all.py",
            "suite_config.json",
            "README.md",
            "requirements.txt",
            "MANIFEST.json",
            "PROVENANCE.json",
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_configs_are_sanitized_and_match_directory(self) -> None:
        for target in TARGETS:
            for config_path in (
                ROOT / target / "model_config.json",
                ROOT / target / "pretrained" / "config.json",
            ):
                config = json.loads(config_path.read_text(encoding="utf-8"))
                self.assertEqual(config["target"], target)
                rendered = json.dumps(config, ensure_ascii=False)
                self.assertNotRegex(rendered, r"[A-Za-z]:[\\/]")
                self.assertNotIn("metrics", config)
                self.assertNotIn("seconds_total", config)

    def test_release_excludes_results_analysis_and_caches(self) -> None:
        forbidden_suffixes = {
            ".csv", ".log", ".png", ".jpg", ".jpeg", ".svg", ".pdf", ".npy", ".npz", ".pyc"
        }
        forbidden_tokens = {"metrics", "predictions", "summary", "analysis", "shap", "plot", "figure"}
        offending = []
        for path in ROOT.rglob("*"):
            if path.is_file():
                name = path.name.lower()
                if path.suffix.lower() in forbidden_suffixes or any(token in name for token in forbidden_tokens):
                    offending.append(path.relative_to(ROOT).as_posix())
            elif path.is_dir() and path.name == "__pycache__":
                offending.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(offending, [])


if __name__ == "__main__":
    unittest.main()
