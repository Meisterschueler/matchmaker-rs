import contextlib
import io
import unittest
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import numpy as np

from report import (
    outerhmm_p95_parity,
    main,
    p95_parity,
    position_parity,
    regression_status,
    renderer_accuracy_parity,
)


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

    def test_outerhmm_p95_limit_and_missing_metrics(self):
        self.assertTrue(p95_parity(0.10, 0.15))
        self.assertFalse(p95_parity(0.10, 0.151))
        self.assertFalse(p95_parity(None, 0.0))

    def test_outerhmm_p95_gate_is_audio_only(self):
        self.assertIsNone(outerhmm_p95_parity("outerhmm", False, 3.081, 0.0))
        self.assertFalse(outerhmm_p95_parity("outerhmm", True, 0.10, 0.151))
        self.assertIsNone(outerhmm_p95_parity("arzt", True, 0.10, 0.151))


class PfKorzReportTests(unittest.TestCase):
    def test_only_pooled_metrics_gate_and_per_seed_results_remain_informational(self):
        pooled = {
            "median_b": 0.02,
            "mean_b": 0.03,
            "accuracy_0.05b": 0.9,
            "accuracy_0.1b": 0.95,
            "accuracy_0.3b": 1.0,
            "accuracy_0.5b": 1.0,
            "accuracy_1b": 1.0,
            "accuracy_2b": 1.0,
        }
        seed_metrics = {
            "1984": {
                "evaluation": {**pooled, "median_b": 4.0, "accuracy_0.1b": 0.0},
                "p95_b": 5.0,
            }
        }
        records = []
        for implementation in ("rust", "python"):
            records.append(
                {
                    "implementation": implementation,
                    "method": "pfkorz",
                    "dataset": "pf-test",
                    "piece": "fixture",
                    "input_type": "audio",
                    "processor": "korzeniowski",
                    "latency_ms": {"feature": {"median": 1.0}, "inference": {"median": 1.0}},
                    "total_processing_s": 1.0,
                    "real_time_factor": 0.1,
                    "evaluation": pooled,
                    "p95_b": 0.05,
                    "ground_truth": True,
                    "ground_truth_path": "ground_truth.tsv",
                    "per_seed_metrics": seed_metrics,
                    "pooled_evaluation": pooled,
                    "pooled_p95_b": 0.05,
                }
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rust_path, python_path = root / "rust.json", root / "python.json"
            rust_path.write_text(json.dumps([records[0]]), encoding="utf-8")
            python_path.write_text(json.dumps([records[1]]), encoding="utf-8")
            output = root / "report"
            with patch(
                "sys.argv",
                ["report.py", "--rust", str(rust_path), "--python", str(python_path), "--out", str(output)],
            ), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    main()

            self.assertEqual(result.exception.code, 0)
            row = json.loads((output / "report.json").read_text(encoding="utf-8"))["rows"][0]
            self.assertNotIn("per_seed_parity", row)
            self.assertIn("1984", row["per_seed_metrics"]["rust"])
            index = (output / "index.md").read_text(encoding="utf-8")
            self.assertIn("only pooled metrics gate PF delivery", index)
            self.assertNotIn("within limits", index)


class PfKorzComparatorTests(unittest.TestCase):
    def test_pf_export_checks_inputs_model_state_and_reports_first_difference(self):
        path = Path(__file__).parents[1] / "golden" / "compare.py"
        spec = importlib.util.spec_from_file_location("pfkorz_comparison", path)
        comparison = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(comparison)
        golden = Path(__file__).parents[2] / "tests" / "golden" / "audio" / "pfkorz"
        metadata = json.loads((golden / "meta.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            exported = Path(temporary)
            processor = exported / "processor/korzeniowski"
            model = exported / "score/pfkorz"
            method = exported / "pfkorz"
            for directory in (processor, model, method):
                directory.mkdir(parents=True)

            for source, target in (
                ("performance_spectrum.npy", processor / "spectrum.npy"),
                ("performance_onset.npy", processor / "onset.npy"),
                ("performance_loudness.npy", processor / "loudness.npy"),
                ("performance_times.npy", processor / "times.npy"),
            ):
                np.save(target, np.load(golden / source))
            for name in (
                "beat_grid.npy",
                "onset_positions.npy",
                "nearest_onsets.npy",
                "rest_mask.npy",
                "templates.npy",
                "active_note_offsets.npy",
                "active_notes.npy",
            ):
                np.save(model / name, np.load(golden / "score_model" / name))

            frames, particles = metadata["frames"], metadata["num_particles"]
            state = {
                "particle_positions": np.zeros((frames, particles)),
                "particle_weights": np.full((frames, particles), 1.0 / particles),
                "note_log_tempi": np.zeros((frames, particles)),
                "local_log_tempi": np.zeros((frames, particles)),
                "sound_probabilities": np.zeros((frames, particles)),
                "rest_probabilities": np.zeros((frames, particles)),
            }
            for name, values in state.items():
                np.save(method / f"{name}.npy", values)
            np.save(method / "effective_sample_size.npy", np.full(frames, particles))
            np.save(method / "resampled.npy", np.zeros(frames, dtype=bool))
            reference_path = np.load(golden / "seed-1984" / "alignment_path.npy")
            np.save(method / "alignment_path.npy", reference_path)
            np.save(method / "positions.npy", reference_path[1])

            (exported / "audio").mkdir()
            (exported / "score").mkdir(exist_ok=True)
            (exported / "audio/meta.json").write_text(
                json.dumps(
                    {
                        "processor": "korzeniowski",
                        "sample_rate": metadata["sample_rate"],
                        "hop_length": metadata["hop_length"],
                        "win_length": metadata["win_length"],
                        "n_fft": metadata["n_fft"],
                        "frames": frames,
                    }
                ),
                encoding="utf-8",
            )
            (exported / "score/meta.json").write_text(
                json.dumps({"audio_method": "pfkorz", "score_model": "pfkorz"}),
                encoding="utf-8",
            )
            (model / "meta.json").write_text(
                json.dumps(
                    {
                        "beat_grid_resolution": 0.1,
                        "frames": len(np.load(model / "beat_grid.npy")),
                        "spectrum_bins": np.load(model / "templates.npy").shape[1],
                    }
                ),
                encoding="utf-8",
            )
            method_meta_path = method / "meta.json"
            method_meta = {
                "config": {
                    "seed": 1984,
                    "num_particles": particles,
                    "notated_tempo": metadata["tempo"],
                    "hop_size": metadata["hop_length"] / metadata["sample_rate"],
                },
                "particle_state_exported": True,
                "particle_state_truncated": False,
            }
            method_meta_path.write_text(json.dumps(method_meta), encoding="utf-8")

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(comparison.compare_pfkorz_case(golden, exported, metadata), 0)
            self.assertIn("WITHIN_TOL", output.getvalue())

            onset = np.load(processor / "onset.npy")
            onset[0] += 1.0
            np.save(processor / "onset.npy", onset)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(comparison.compare_pfkorz_case(golden, exported, metadata), 1)
            self.assertIn("first differing stage: processor/korzeniowski/onset.npy", output.getvalue())

            np.save(processor / "onset.npy", np.load(golden / "performance_onset.npy"))
            method_meta["particle_state_exported"] = False
            method_meta["particle_state_truncated"] = True
            method_meta_path.write_text(json.dumps(method_meta), encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(comparison.compare_pfkorz_case(golden, exported, metadata), 1)
            self.assertIn("particle_state_exported", output.getvalue())


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


class MelStageComparisonTests(unittest.TestCase):
    def test_features_report_first_frame_and_band_and_check_metadata(self):
        path = Path(__file__).parents[1] / "golden" / "compare.py"
        spec = importlib.util.spec_from_file_location("mel_stage_comparison", path)
        comparison = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(comparison)
        metadata = {
            "processor": "mel",
            "sample_rate": 44_100,
            "hop_length": 1_470,
            "n_fft": 2_940,
            "n_mels": 32,
            "norm": "inf",
            "frames": 2,
            "tolerances": {"features_rtol": 1e-5, "features_atol": 1e-5},
        }
        reference = np.zeros((2, 32), dtype=np.float32)
        reference[0, 0] = 100.0
        times = np.array([0.0, 0.1])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            golden, exported = root / "golden", root / "export"
            golden.mkdir()
            np.save(golden / "features.npy", reference)
            np.save(golden / "times.npy", times)
            (exported / "processor/mel").mkdir(parents=True)
            (exported / "audio").mkdir()
            np.save(exported / "processor/mel/times.npy", times)
            export_meta = {
                key: value
                for key, value in metadata.items()
                if key != "tolerances"
            }
            (exported / "audio/meta.json").write_text(json.dumps(export_meta))

            def run(features):
                np.save(exported / "processor/mel/features.npy", features)
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    return comparison.compare_mel_case(golden, exported, metadata), output.getvalue()

            within = reference.copy()
            within[1, 5] = 9e-6
            self.assertEqual(run(within)[0], 0)
            beyond = reference.copy()
            beyond[1, 5] = 2e-5
            status, message = run(beyond)
            self.assertEqual(status, 1)
            self.assertIn("frame 1, mel band 5", message)

            np.save(exported / "processor/mel/features.npy", reference)
            export_meta["n_mels"] = 64
            (exported / "audio/meta.json").write_text(json.dumps(export_meta))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(comparison.compare_mel_case(golden, exported, metadata), 1)
            self.assertIn("n_mels", output.getvalue())

    def test_audio_position_gate_is_shared_by_frame_processors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rust, python = root / "rust.tsv", root / "python.tsv"
            times = np.arange(3, dtype=float) / 30
            expected = np.column_stack((times, np.zeros(3)))
            actual = expected.copy()
            actual[1, 1] = 0.5
            np.savetxt(rust, actual, delimiter="\t", header="perf_sec\tscore_beat", comments="")
            np.savetxt(python, expected, delimiter="\t", header="perf_sec\tscore_beat", comments="")
            result = position_parity("arzt", rust, python, audio=True)
            self.assertTrue(result["ok"])
            self.assertEqual(result["class"], "WITHIN_TOL")


class DannenbergPositionParityTests(unittest.TestCase):
    def test_equal_dp_values_do_not_hide_position_difference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rust, python = root / "rust.tsv", root / "python.tsv"
            np.savetxt(rust, [[0.0, 0.0], [1.0, 2.0]], delimiter="\t", header="time\tbeat", comments="")
            np.savetxt(python, [[0.0, 0.0], [1.0, 3.0]], delimiter="\t", header="time\tbeat", comments="")
            np.savetxt(rust.with_suffix(".gap.tsv"), [1.0, 0.0])

            self.assertFalse(position_parity("dannenberg", rust, python)["ok"])
            self.assertTrue(position_parity("pthmm", rust, python)["ok"])

    def test_report_discloses_disabled_numpy_cpu_features(self):
        record = {
            "method": "dannenberg", "dataset": "fixture", "piece": "test",
            "processor": "pitch", "evaluation": None, "total_processing_s": 1.0,
            "environment": {"numpy_disabled_cpu_features": "X86_V3,X86_V4"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rust, python = root / "rust.json", root / "python.json"
            rust.write_text(json.dumps([{**record, "environment": {}}]), encoding="utf-8")
            python.write_text(json.dumps([record]), encoding="utf-8")
            output = root / "report"
            with patch("sys.argv", [
                "report.py", "--rust", str(rust), "--python", str(python), "--out", str(output),
            ]), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    main()
            self.assertEqual(result.exception.code, 0)
            self.assertIn("X86_V3,X86_V4", (output / "index.md").read_text(encoding="utf-8"))
            row = json.loads((output / "report.json").read_text(encoding="utf-8"))["rows"][0]
            self.assertEqual(row["python"]["numpy_disabled_cpu_features"], "X86_V3,X86_V4")


class MfccStageComparisonTests(unittest.TestCase):
    def test_feature_limit_first_difference_and_metadata(self):
        path = Path(__file__).parents[1] / "golden" / "compare.py"
        spec = importlib.util.spec_from_file_location("mfcc_stage_comparison", path)
        comparison = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(comparison)
        metadata = {
            "processor": "mfcc",
            "sample_rate": 44_100,
            "hop_length": 735,
            "n_fft": 1_470,
            "n_mfcc": 8,
            "norm": "ortho",
            "frames": 2,
            "tolerances": {"features_rtol": 1e-5, "features_atol": 2e-4},
        }
        reference = np.zeros((2, 8), dtype=np.float32)
        times = np.array([0.0, 0.1])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            golden, exported = root / "golden", root / "export"
            golden.mkdir()
            np.save(golden / "features.npy", reference)
            np.save(golden / "times.npy", times)
            (exported / "processor/mfcc").mkdir(parents=True)
            (exported / "audio").mkdir()
            export_meta = {
                key: value
                for key, value in metadata.items()
                if key != "tolerances"
            }
            (exported / "audio/meta.json").write_text(json.dumps(export_meta))
            np.save(exported / "processor/mfcc/times.npy", times)

            def run(features):
                np.save(exported / "processor/mfcc/features.npy", features)
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    return comparison.compare_mfcc_case(golden, exported, metadata), output.getvalue()

            within = reference.copy()
            within[1, 5] = 1e-4
            self.assertEqual(run(within)[0], 0)
            beyond = reference.copy()
            beyond[1, 5] = 3e-4
            status, message = run(beyond)
            self.assertEqual(status, 1)
            self.assertIn("frame 1, MFCC coefficient 5", message)

            np.save(exported / "processor/mfcc/features.npy", reference)
            export_meta["n_mfcc"] = 13
            (exported / "audio/meta.json").write_text(json.dumps(export_meta))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(comparison.compare_mfcc_case(golden, exported, metadata), 1)
            self.assertIn("n_mfcc", output.getvalue())


class OuterHmmComparisonTests(unittest.TestCase):
    def test_weighted_score_arrays_must_match_exactly(self):
        path = Path(__file__).parents[1] / "golden" / "compare.py"
        spec = importlib.util.spec_from_file_location("outerhmm_comparison", path)
        comparison = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(comparison)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            golden = root / "golden" / "weighted"
            base = golden.parent
            exported = root / "export"
            model = exported / "score/outerhmm"
            golden.mkdir(parents=True)
            model.mkdir(parents=True)

            template_meta = {
                "archive_sha256": "archive",
                "license": "Apache-2.0",
                "upstream_commit": "commit",
                "training_corpus": "unknown",
            }
            export_meta = {
                "pitch_template_archive_sha256": "archive",
                "pitch_template_license": "Apache-2.0",
                "pitch_template_upstream_commit": "commit",
                "pitch_template_training_corpus": "unknown",
                "soundfont_sha256": "soundfont",
            }
            (base / "template_meta.json").write_text(json.dumps(template_meta))
            (exported / "score/meta.json").write_text(json.dumps(export_meta))
            support = np.zeros(88, dtype=bool)
            support[0] = True
            templates = np.ones((88, 88), dtype=np.float64)
            offsets = np.array([0, 1])
            pitches = np.array([60, 64])
            values = np.array([0.5, 0.5])
            for name, value in (
                ("template_support.npy", support),
                ("learned_pitch_templates.npy", templates),
                ("pitch_templates.npy", templates),
            ):
                np.save(base / name, value)
            for name, value in (
                ("score_positions.npy", np.array([0, 1])),
                ("weighted_offsets.npy", offsets),
                ("weighted_pitch.npy", pitches),
                ("weighted_value.npy", values),
                ("chord_templates.npy", templates[:2]),
            ):
                np.save(golden / name, value)
                np.save(model / name, value)
            np.save(model / "learned_support.npy", support)
            np.save(model / "pitch_templates.npy", templates)
            np.save(model / "chord_templates.npy", templates[:2])

            metadata = {
                "soundfont_sha256": "soundfont",
                "tolerances": {"model_rtol": 0.0, "model_atol": 0.0},
            }
            self.assertEqual(comparison.compare_outerhmm_case(golden, exported, metadata), 0)
            np.save(model / "weighted_offsets.npy", np.array([0, 2]))
            self.assertEqual(comparison.compare_outerhmm_case(golden, exported, metadata), 1)


if __name__ == "__main__":
    unittest.main()
