"""Audio `arzt` facade parity and validation tests."""

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


def make_audio(method="arzt", **kwargs):
    return Matchmaker(
        score_file=str(SCORE),
        performance_file=str(PERFORMANCE),
        input_type="audio",
        method=method,
        kwargs={"soundfont_path": str(SOUNDFONT), **kwargs},
    )


def test_audio_arzt_defaults_yield_positions_and_return_path():
    assert matchmaker_rs.AVAILABLE_METHODS["audio"] == ["arzt", "dixon", "arzt_tempo"]
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
