import contextlib
import io
import unittest
import importlib.util
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import numpy as np

from report import regression_status, renderer_accuracy_parity


class RegressionStatusTests(unittest.TestCase):
    def test_missing_baseline_is_unknown(self):
        self.assertIsNone(regression_status(1.0, None))

    def test_latency_over_ten_percent_is_regression(self):
        self.assertTrue(regression_status(1.11, 1.0))

    def test_latency_at_ten_percent_is_not_regression(self):
        self.assertFalse(regression_status(1.1, 1.0))

    def test_renderer_accuracy_limit_and_missing_metrics(self):
        expected = {key: 1.0 for key in (
            "median_b", "accuracy_0.05b", "accuracy_0.1b", "accuracy_0.3b",
            "accuracy_0.5b", "accuracy_1b", "accuracy_2b",
        )}
        self.assertTrue(renderer_accuracy_parity({**expected, "accuracy_0.05b": 0.99}, expected))
        self.assertFalse(renderer_accuracy_parity({**expected, "accuracy_0.05b": 0.98}, expected))
        self.assertFalse(renderer_accuracy_parity({}, expected))


class TempoStageComparisonTests(unittest.TestCase):
    def test_declared_metric_limit_is_enforced(self):
        path = Path(__file__).parents[1] / "golden" / "compare.py"
        spec = importlib.util.spec_from_file_location("stage_comparison", path)
        comparison = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(comparison)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            golden, exported = root / "golden", root / "export"
            golden.mkdir()
            times = np.arange(3, dtype=float) / 30
            for name, relative, values in (
                ("performance_features.npy", "processor/chroma/features.npy", np.ones((3, 1))),
                ("performance_times.npy", "processor/chroma/times.npy", times),
                ("score_features.npy", "score/audio/features.npy", np.ones((3, 1))),
                ("score_frame_to_beat.npy", "score/audio/frame_to_beat.npy", np.zeros(3)),
                ("score_positions.npy", "score/audio/score_positions.npy", np.zeros(1)),
                ("alignment_path.npy", "arzt_tempo/alignment_path.npy", np.array([times, np.zeros(3)])),
            ):
                np.save(golden / name, values)
                target = exported / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                np.save(target, values)
            np.save(exported / "arzt_tempo/tempo_states.npy", np.zeros((3, 8)))
            metrics = {"median_b": 0.0, "accuracy_0.05b": 0.97}
            evaluation = ModuleType("matchmaker.utils.eval")
            modules = {
                "matchmaker": ModuleType("matchmaker"),
                "matchmaker.utils": ModuleType("matchmaker.utils"),
                "matchmaker.utils.eval": evaluation,
            }
            metadata = {"method": "arzt_tempo", "processor": "chroma", "frame_rate": 30}
            with patch.dict(sys.modules, modules):
                evaluation.evaluate_alignment = Mock(side_effect=[metrics, metrics])
                self.assertEqual(comparison.compare_tempo_audio_case(golden, exported, metadata), 0)
                evaluation.evaluate_alignment = Mock(side_effect=[metrics, {**metrics, "accuracy_0.05b": 0.90}])
                self.assertEqual(comparison.compare_tempo_audio_case(golden, exported, metadata), 1)


class CqtStageComparisonTests(unittest.TestCase):
    def test_frame_maximum_term_widens_only_the_cqt_feature_limit(self):
        path = Path(__file__).parents[1] / "golden" / "compare.py"
        spec = importlib.util.spec_from_file_location("cqt_stage_comparison", path)
        comparison = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(comparison)
        tolerances = {"features_atol": 1e-5, "features_rtol": 1e-5, "features_frame_max_rtol": 1e-5}
        metadata = {
            "processor": "cqt", "sample_rate": 44_100, "hop_length": 1_470,
            "stream_frames": 2, "tolerances": tolerances,
        }
        reference = np.zeros((2, 88), dtype=np.float32)
        reference[:, 0] = 100.0  # frame maximum 100: a zero bin may deviate by 1e-5 + 1e-3
        times = np.array([0.0, 0.1])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            golden, exported = root / "golden", root / "export"
            golden.mkdir()
            np.save(golden / "stream_features.npy", reference)
            np.save(golden / "stream_times.npy", times)
            (exported / "processor/cqt").mkdir(parents=True)
            (exported / "audio").mkdir()
            np.save(exported / "processor/cqt/times.npy", times)
            (exported / "audio/meta.json").write_text(
                '{"processor": "cqt", "sample_rate": 44100, "hop_length": 1470, "frames": 2}'
            )

            def run(features):
                np.save(exported / "processor/cqt/features.npy", features)
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    return comparison.compare_cqt_case(golden, exported, metadata), output.getvalue()

            within = reference.copy()
            within[1, 5] = 5e-4
            self.assertEqual(run(within)[0], 0)
            beyond = reference.copy()
            beyond[1, 5] = 2e-3
            status, message = run(beyond)
            self.assertEqual(status, 1)
            self.assertIn("frame 1, bin 5", message)
            self.assertEqual(run(reference[:1])[0], 1)


if __name__ == "__main__":
    unittest.main()
