"""Audio `arzt` facade parity and validation tests."""

import json
from pathlib import Path

import numpy as np
import pytest

import matchmaker_rs
from matchmaker_rs import Matchmaker

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "tests" / "data"
GOLDEN = ROOT / "tests" / "golden" / "audio"
SCORE = DATA / "simple_mozart_k265_var1.musicxml"
PERFORMANCE = DATA / "audio" / "simple_mozart_k265_var1_sf2.wav"
SOUNDFONT = (
    ROOT
    / "target"
    / "audio-assets"
    / "UprightPianoKW-small-SF2-20190703"
    / "UprightPianoKW-small-20190703.sf2"
)


def make_audio(
    method="arzt",
    *,
    score=SCORE,
    performance=PERFORMANCE,
    unfold_score=True,
    **kwargs,
):
    return Matchmaker(
        score_file=str(score),
        performance_file=str(performance),
        input_type="audio",
        method=method,
        kwargs={"soundfont_path": str(SOUNDFONT), **kwargs},
        unfold_score=unfold_score,
    )


def test_audio_arzt_defaults_yield_positions_and_return_path():
    assert matchmaker_rs.AVAILABLE_METHODS["audio"] == [
        "arzt", "dixon", "arzt_tempo", "arzt_multi_fold", "skf", "outerhmm"
    ]
    defaults = matchmaker_rs.DEFAULT_KWARGS["audio"]["arzt"]
    assert defaults["processor"] == "chroma"
    assert defaults["frame_rate"] == 30
    assert defaults["step_size"] == 3

    mm = make_audio()
    assert (mm.input_type, mm.method) == ("audio", "arzt")
    run = mm.run(verbose=False)
    positions = []
    while True:
        try:
            positions.append(next(run))
        except StopIteration as stop:
            returned_path = stop.value
            break

    expected = np.load(GOLDEN / "arzt_chroma" / "positions.npy")
    expected_path = np.load(GOLDEN / "arzt_chroma" / "alignment_path.npy")
    assert len(positions) == 226
    assert all(type(position) is float for position in positions)
    np.testing.assert_allclose(positions, expected, rtol=0, atol=0.3)
    assert returned_path.dtype == np.float64 and returned_path.shape == (2, 226)
    np.testing.assert_allclose(returned_path[0], expected_path[0], rtol=0, atol=1e-6)
    np.testing.assert_allclose(returned_path[1], expected_path[1], rtol=0, atol=0.3)
    np.testing.assert_array_equal(mm.score_follower.alignment_path, returned_path)
    assert mm._has_run


def test_audio_arzt_tempo_defaults_seed_and_path():
    assert "arzt_tempo" in matchmaker_rs.AVAILABLE_METHODS["audio"]
    defaults = matchmaker_rs.DEFAULT_KWARGS["audio"]["arzt_tempo"]
    assert defaults["processor"] == "chroma"
    assert defaults["frame_rate"] == 30
    assert defaults["step_size"] == 3
    default = make_audio("arzt_tempo")
    explicit = make_audio("arzt_tempo", random_seed=1984)
    positions = list(default.run(verbose=False))
    assert positions and all(type(position) is float for position in positions)
    assert positions == list(explicit.run(verbose=False))
    assert default.score_follower.alignment_path.shape == (2, len(positions))
    np.testing.assert_array_equal(
        default.score_follower.alignment_path,
        explicit.score_follower.alignment_path,
    )
    with pytest.raises(ValueError, match="random_seed"):
        make_audio("arzt_tempo", random_seed=-1)


def test_audio_arzt_multi_fold_defaults_and_repeat_mode():
    assert "arzt_multi_fold" in matchmaker_rs.AVAILABLE_METHODS["audio"]
    defaults = matchmaker_rs.DEFAULT_KWARGS["audio"]["arzt_multi_fold"]
    assert defaults["selection_window"] == 50
    assert defaults["switch_margin"] == 0.05

    mm = make_audio(
        "arzt_multi_fold",
        score=DATA / "fixture_repeats_voltas.musicxml",
        performance=DATA / "audio/arzt_multi_fold/repeat_taken.wav",
        unfold_score=False,
    )
    positions = list(mm.run(verbose=False))
    assert positions and all(type(position) is float for position in positions)
    assert all(0.0 <= position <= 14.0 for position in positions)
    assert mm.score_follower.alignment_path.shape == (2, len(positions))

    with pytest.raises(ValueError, match="selection_window"):
        make_audio("arzt_multi_fold", selection_window=0)
    with pytest.raises(ValueError, match="selection_window"):
        make_audio("arzt_multi_fold", selection_window=None)
    with pytest.raises(ValueError, match="switch_margin"):
        make_audio("arzt_multi_fold", switch_margin=1.1)
    with pytest.raises(ValueError, match="switch_margin"):
        make_audio("arzt_multi_fold", switch_margin=None)


def test_audio_arzt_reports_typed_configuration_and_file_errors():
    with pytest.raises(ValueError, match="soundfont_path"):
        Matchmaker(
            score_file=str(SCORE),
            performance_file=str(PERFORMANCE),
            input_type="audio",
            method="arzt",
        )

    with pytest.raises(ValueError, match="step_size"):
        make_audio(step_size=0)

    with pytest.raises(OSError, match="missing.wav"):
        mm = Matchmaker(
            score_file=str(SCORE),
            performance_file=str(DATA / "missing.wav"),
            input_type="audio",
            method="arzt",
            kwargs={"soundfont_path": str(SOUNDFONT)},
        )
        list(mm.run(verbose=False))


def test_audio_dixon_defaults_yield_positions_and_allow_processor_override():
    defaults = matchmaker_rs.DEFAULT_KWARGS["audio"]["dixon"]
    assert defaults["processor"] == "lse"
    assert defaults["frame_rate"] == 50
    assert defaults["max_run_count"] == 3

    for processor, kwargs, tolerance in [
        ("lse", {}, 0.5),
        ("chroma", {"processor": "chroma"}, 0.3),
    ]:
        mm = make_audio("dixon", **kwargs)
        run = mm.run(verbose=False)
        positions = []
        while True:
            try:
                positions.append(next(run))
            except StopIteration as stop:
                returned_path = stop.value
                break

        expected = np.load(GOLDEN / f"dixon_{processor}" / "positions.npy")
        expected_path = np.load(GOLDEN / f"dixon_{processor}" / "alignment_path.npy")
        shared_frames = min(len(positions), len(expected))
        assert abs(len(positions) - len(expected)) <= 1
        np.testing.assert_allclose(
            positions[:shared_frames], expected[:shared_frames], rtol=0, atol=tolerance
        )
        assert returned_path.dtype == np.float64 and returned_path.shape[0] == 2
        assert abs(returned_path.shape[1] - expected_path.shape[1]) <= 1
        shared_path_frames = min(returned_path.shape[1], expected_path.shape[1])
        np.testing.assert_allclose(
            returned_path[0, :shared_path_frames],
            expected_path[0, :shared_path_frames],
            rtol=0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            returned_path[1, :shared_path_frames],
            expected_path[1, :shared_path_frames],
            rtol=0,
            atol=0.6,
        )
        np.testing.assert_array_equal(mm.score_follower.alignment_path, returned_path)
        assert mm._has_run

    with pytest.raises(ValueError, match="max_run_count"):
        make_audio("dixon", max_run_count=0)


def test_audio_arzt_lse_override_uses_l2_default():
    mm = make_audio("arzt", processor="lse")
    positions = list(mm.run(verbose=False))
    expected = np.load(GOLDEN / "arzt_lse" / "positions.npy")
    np.testing.assert_allclose(positions, expected, rtol=0, atol=0.3)


@pytest.mark.parametrize("method,processor", [("arzt", "chroma"), ("dixon", "lse")])
def test_audio_export_writes_pipeline_stages_without_consuming_run(method, processor, tmp_path):
    mm = make_audio(method, processor=processor)
    mm.export(tmp_path)
    for name in [
        "audio/meta.json",
        "audio/frame_times.npy",
        f"processor/{processor}/features.npy",
        f"processor/{processor}/times.npy",
        "score/audio/features.npy",
        "score/audio/frame_to_beat.npy",
        f"{method}/observations_features.npy",
        f"{method}/positions.npy",
        f"{method}/current_index.npy",
        f"{method}/alignment_path.npy",
    ]:
        assert (tmp_path / name).is_file(), f"missing export stage {name}"
    assert not mm._has_run


def test_raw_spectrum_extraction_returns_features_and_times():
    features, times = matchmaker_rs.extract_audio_features(
        str(PERFORMANCE), "raw_spectrum"
    )
    assert features.dtype == np.float32
    assert features.shape[1] == 257
    assert times.dtype == np.float64
    assert times.shape == (features.shape[0],)

    overridden, overridden_times = matchmaker_rs.extract_audio_features(
        str(PERFORMANCE),
        "raw_spectrum",
        {"sample_rate": 16_000, "hop_length": 256, "n_fft": 1_024},
    )
    assert overridden.shape[1] == 513
    assert overridden_times.shape == (overridden.shape[0],)
    with pytest.raises(ValueError, match="n_fft"):
        matchmaker_rs.extract_audio_features(
            str(PERFORMANCE),
            "raw_spectrum",
            {"hop_length": 256, "n_fft": 128},
        )


def test_raw_spectrum_audio_following_with_compatible_hop():
    mm = make_audio(
        "arzt",
        processor="raw_spectrum",
        sample_rate=44_100,
        hop_length=128,
        n_fft=512,
    )
    positions = list(mm.run(verbose=False))
    assert positions and all(np.isfinite(position) for position in positions)
    assert mm.score_follower.alignment_path.shape == (2, len(positions))


def test_cqt_extraction_returns_88_bins_and_rejects_foreign_options():
    features, times = matchmaker_rs.extract_audio_features(str(PERFORMANCE), "cqt")
    assert features.dtype == np.float32
    assert features.shape == (227, 88)
    assert np.isfinite(features).all() and (features >= 0).all()
    assert times.dtype == np.float64
    assert times.shape == (227,)

    overridden, _ = matchmaker_rs.extract_audio_features(
        str(PERFORMANCE), "cqt", {"sample_rate": 44_100, "hop_length": 735}
    )
    assert overridden.shape[1] == 88 and overridden.shape[0] > features.shape[0]
    with pytest.raises(ValueError, match="n_fft"):
        matchmaker_rs.extract_audio_features(str(PERFORMANCE), "cqt", {"n_fft": 512})
    with pytest.raises(ValueError, match="sample_rate"):
        matchmaker_rs.extract_audio_features(str(PERFORMANCE), "cqt", {"sample_rate": 8_000})
    with pytest.raises(ValueError, match="unknown audio processor"):
        matchmaker_rs.extract_audio_features(str(PERFORMANCE), "mel")


@pytest.mark.parametrize("method,steps", [("arzt", 226), ("dixon", 379)])
def test_cqt_audio_following_yields_the_reference_step_count(method, steps):
    mm = make_audio(method, processor="cqt")
    positions = list(mm.run(verbose=False))
    assert len(positions) == steps
    assert all(np.isfinite(position) for position in positions)
    assert mm.score_follower.alignment_path.shape == (2, steps)


def test_cqt_rejects_incompatible_methods_and_filter_norms():
    for method in ["arzt_tempo", "arzt_multi_fold", "skf"]:
        with pytest.raises(ValueError, match="processor"):
            make_audio(method, processor="cqt")
    with pytest.raises(ValueError, match="norm"):
        make_audio("arzt", processor="cqt", norm=2.0)


def test_cqt_spectral_flux_extraction_returns_89_or_88_columns():
    features, times = matchmaker_rs.extract_audio_features(str(PERFORMANCE), "cqt_spectral_flux")
    assert features.dtype == np.float32
    assert features.shape == (227, 89)
    assert np.isfinite(features).all() and (features >= 0).all()
    assert features[0, 88] > 0  # the primed first call already set the history
    assert times.shape == (227,)

    plain, _ = matchmaker_rs.extract_audio_features(str(PERFORMANCE), "cqt")
    without, _ = matchmaker_rs.extract_audio_features(
        str(PERFORMANCE), "cqt_spectral_flux", {"include_spectral_flux": False}
    )
    assert without.shape == (227, 88)
    np.testing.assert_array_equal(without, plain)
    np.testing.assert_array_equal(features[:, :88], plain)

    for processor, kwargs in [
        ("cqt_spectral_flux", {"include_spectral_flux": 1}),
        ("cqt", {"include_spectral_flux": False}),
        ("cqt_spectral_flux", {"n_fft": 512}),
        ("cqt_spectral_flux", {"sample_rate": 8_000}),
    ]:
        with pytest.raises(ValueError, match=next(iter(kwargs))):
            matchmaker_rs.extract_audio_features(str(PERFORMANCE), processor, kwargs)


def test_cqt_spectral_flux_has_no_frame_follower():
    for method in ["arzt", "dixon", "arzt_tempo", "arzt_multi_fold", "skf"]:
        with pytest.raises(ValueError, match="only by audio outerhmm"):
            make_audio(method, processor="cqt_spectral_flux")


def test_audio_outerhmm_defaults_override_and_export(tmp_path):
    assert matchmaker_rs.DEFAULT_KWARGS["audio"]["outerhmm"] == {
        "processor": "cqt_spectral_flux",
        "sample_rate": 16_000,
        "frame_rate": 25,
        "s_j": 0.0,
    }
    mm = make_audio("outerhmm")
    positions = list(mm.run(verbose=False))
    assert positions and all(np.isfinite(position) for position in positions)
    assert mm.score_follower.alignment_path.shape == (2, len(positions))
    mm = make_audio("outerhmm", sample_rate=22_050, hop_length=512)
    override_positions = list(mm.run(verbose=False))
    assert override_positions and all(np.isfinite(position) for position in override_positions)

    exported = make_audio("outerhmm")
    exported.export(tmp_path)
    for name in [
        "processor/cqt_spectral_flux/features.npy",
        "score/outerhmm/pitch_templates.npy",
        "score/outerhmm/learned_support.npy",
        "score/outerhmm/chord_templates.npy",
        "outerhmm/state_probabilities.npy",
        "outerhmm/sound_probabilities.npy",
        "outerhmm/pause_probabilities.npy",
    ]:
        assert (tmp_path / name).is_file(), name
    assert not (tmp_path / "score/audio/features.npy").exists()

    for kwargs, message in [
        ({"processor": "cqt"}, "processor"),
        ({"s_j": 1.1}, "s_j"),
        ({"distance_func": "euclidean"}, "Invalid audio parameter distance_func"),
    ]:
        with pytest.raises(ValueError, match=message):
            make_audio("outerhmm", **kwargs)


def test_midi_outerhmm_remains_available():
    mm = Matchmaker(
        score_file=str(SCORE),
        performance_file=str(DATA / "simple_mozart_k265_var1.mid"),
        input_type="midi",
        method="outerhmm",
        kwargs={},
    )
    positions = list(mm.run(verbose=False))
    assert positions and all(np.isfinite(position) for position in positions)


def test_audio_skf_defaults_template_modes_and_errors():
    assert matchmaker_rs.DEFAULT_KWARGS["audio"]["skf"] == {
        "processor": "raw_spectrum",
        "sample_rate": 8000,
        "hop_length": 128,
        "n_fft": 512,
    }
    mm = make_audio("skf")
    positions = list(mm.run(verbose=False))
    assert positions and all(type(position) is float for position in positions)
    assert mm.score_follower.alignment_path.shape == (2, len(positions))

    gaussian = Matchmaker(
        score_file=str(SCORE),
        performance_file=str(PERFORMANCE),
        input_type="audio",
        method="skf",
        kwargs={"template_mode": "gaussian"},
    )
    gaussian_positions = list(gaussian.run(verbose=False))
    assert gaussian_positions and gaussian_positions != positions, "the mode changes the follower"

    for kwargs, message in [
        ({"template_mode": "auto"}, "template_mode"),
        ({"template_mode": 3}, "template_mode"),
        ({"window_size": 5.0}, "window_size"),
        ({"distance_func": "euclidean"}, "distance_func"),
        ({"processor": "chroma"}, "processor"),
        ({"n_fft": 64}, "n_fft"),
        ({"step_size": 3}, "step_size"),
    ]:
        with pytest.raises(ValueError, match=message):
            make_audio("skf", **kwargs)
    with pytest.raises(ValueError, match="soundfont_path"):
        Matchmaker(
            score_file=str(SCORE),
            performance_file=str(PERFORMANCE),
            input_type="audio",
            method="skf",
            kwargs={},
        )


def test_audio_skf_export_records_template_mode_and_hypothesis_state(tmp_path):
    mm = make_audio("skf")
    mm.export(tmp_path)
    for name in [
        "score/skf/templates.npy",
        "score/skf/chord_offsets.npy",
        "skf/state_offsets.npy",
        "skf/state_probability.npy",
        "skf/positions.npy",
        "skf/alignment_path.npy",
    ]:
        assert (tmp_path / name).is_file(), f"missing export stage {name}"
    assert not (tmp_path / "score/audio/features.npy").exists()
    assert np.load(tmp_path / "score/skf/templates.npy").shape == (54, 257)
    meta = json.loads((tmp_path / "score/meta.json").read_text())
    assert meta["template_mode"] == "synthesized"
    assert meta["soundfont"].endswith("UprightPianoKW-small-20190703.sf2")
    assert not mm._has_run
