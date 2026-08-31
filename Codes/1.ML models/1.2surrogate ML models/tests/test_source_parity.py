from __future__ import annotations

import hashlib
import json
import pickle
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from surrogate_core.constants import TARGETS
from surrogate_core.fingerprint import featurize_hashed_morgan_counts
from surrogate_core.inference import predict_smiles


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SourceParityTests(unittest.TestCase):
    def test_provenance_manifest_matches_source_and_release_artifacts(self) -> None:
        provenance_path = ROOT / "PROVENANCE.json"
        self.assertTrue(provenance_path.is_file())
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        self.assertEqual(len(provenance["artifacts"]), 24)
        if not all((SOURCE_ROOT / target).is_dir() for target in TARGETS):
            self.skipTest("Sibling source artifact folders are not available in this relocated release.")
        for item in provenance["artifacts"]:
            source = SOURCE_ROOT / item["target"] / item["filename"]
            released = ROOT / item["release_path"]
            self.assertEqual(sha256(source), item["source_sha256"])
            self.assertEqual(sha256(released), item["release_sha256"])
            self.assertEqual(item["source_sha256"], item["release_sha256"])

    def test_source_and_release_predictions_match_on_fixed_smiles(self) -> None:
        if not all((SOURCE_ROOT / target).is_dir() for target in TARGETS):
            self.skipTest("Sibling source artifact folders are not available in this relocated release.")
        try:
            import tensorflow as tf
        except ImportError:
            self.skipTest("TensorFlow is not installed.")
        smiles = ["COCOC", "CC(OC)OCC"]
        values, valid = featurize_hashed_morgan_counts(smiles)
        self.assertTrue(valid.all())
        for target in TARGETS:
            with self.subTest(target=target):
                released, _ = predict_smiles(smiles, target, ROOT, batch_size=64)
                source_model = tf.keras.models.load_model(
                    SOURCE_ROOT / target / "electronic_model.keras", compile=False
                )
                with (SOURCE_ROOT / target / "Y_scaler.pkl").open("rb") as handle:
                    y_scaler = pickle.load(handle)
                source_scaled = source_model.predict(values, batch_size=64, verbose=0).reshape(-1, 1)
                expected = y_scaler.inverse_transform(source_scaled).reshape(-1)
                self.assertEqual(float(abs(released - expected).max()), 0.0)
                tf.keras.backend.clear_session()


if __name__ == "__main__":
    unittest.main()
