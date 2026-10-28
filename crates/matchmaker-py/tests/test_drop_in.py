"""Drop-in behaviour of ``matchmaker_rs.Matchmaker`` (T062): README quickstart pattern,
reference-shaped attributes and errors, and equality with the golden data of pymatchmaker."""

from pathlib import Path

import numpy as np
import pytest

import matchmaker_rs
from matchmaker_rs import Matchmaker

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "tests" / "data"
GOLDEN = ROOT / "tests" / "golden"
PIECE = "simple_mozart_k265_var1"
SCORE = DATA / f"{PIECE}.musicxml"
PERFORMANCE = DATA / f"{PIECE}.mid"
METHODS = ["pthmm", "hmm", "outerhmm"]


def make(method="pthmm", **kwargs):
    return Matchmaker(
        score_file=str(SCORE),
        performance_file=str(PERFORMANCE),
        input_type="midi",
        method=method,
        **kwargs,
    )


def golden(method, name, piece=PIECE):
    return np.load(GOLDEN / piece / method / name)


@pytest.mark.parametrize("method", METHODS)
def test_quickstart_matches_reference(method):
    mm = make(method)
    positions = [beat for beat in mm.run(verbose=False)]
    assert all(type(beat) is float for beat in positions)
    np.testing.assert_allclose(positions, golden(method, "positions.npy"), rtol=0, atol=1e-6)

    path = mm.score_follower.alignment_path
    assert path.dtype == np.float64 and path.shape[0] == 2
    np.testing.assert_allclose(path, golden(method, "alignment_path.npy"), rtol=0, atol=1e-6)
    assert mm.score_follower._alignment_path == [tuple(c) for c in path.T.tolist()]


@pytest.mark.parametrize("method", METHODS)
def test_positions_are_the_states(method):
    mm = make(method)
    follower_states = mm.score_follower.score_positions
    assert follower_states.dtype == np.float32
    np.testing.assert_array_equal(follower_states, golden(method, "score_positions.npy"))

    note_array = np.load(GOLDEN / PIECE / "score" / "note_array.npy")
    np.testing.assert_array_equal(mm.score_positions, np.unique(note_array["onset_beat"]))
    assert mm.score_positions.dtype == np.float32


def test_run_returns_the_alignment_path():
    mm = make()
    run = mm.run(verbose=False)
    with pytest.raises(StopIteration) as stop:
        while True:
            next(run)
    np.testing.assert_array_equal(stop.value.value, mm.score_follower.alignment_path)
    assert mm._has_run


def test_run_is_lazy_and_single_use():
    mm = make()
    run = mm.run(verbose=False)
    assert mm.score_follower.alignment_path.shape == (2, 0)
    next(run)
    assert mm.score_follower.alignment_path.shape[1] >= 1
    with pytest.raises(RuntimeError):
        mm.run(verbose=False)


def test_attributes_and_module_constants():
    mm = make("outerhmm")
    assert (mm.method, mm.input_type, mm.tempo) == ("outerhmm", "midi", 120.0)
    assert mm.polling_period == 0.001
    assert mm.config == {"piano_range": True}
    assert make().polling_period == 0.01
    assert matchmaker_rs.AVAILABLE_METHODS == {"midi": METHODS}
    assert matchmaker_rs.DEFAULT_KWARGS["midi"]["outerhmm"]["polling_period"] == 0.001
    assert matchmaker_rs.__reference_version__ == "v0.4.0"
    assert isinstance(matchmaker_rs.__version__, str)
    assert make(tempo=90).tempo == 90.0
    assert Matchmaker(SCORE, PERFORMANCE, "midi").method == "pthmm"


def test_pathlike_arguments():
    mm = Matchmaker(SCORE, PERFORMANCE, "midi", "pthmm")
    assert mm.score_file == str(SCORE)


@pytest.mark.parametrize("kwargs", [{"method": "arzt"}, {"input_type": "audio"}])
def test_unsupported_method_or_input_type(kwargs):
    arguments = {"score_file": str(SCORE), "performance_file": str(PERFORMANCE), "input_type": "midi"}
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=r"\['pthmm', 'hmm', 'outerhmm'\]"):
        Matchmaker(**arguments)


def test_file_errors_follow_the_reference():
    with pytest.raises(ValueError, match="Expected MIDI file, but got .*performance.wav"):
        Matchmaker(str(SCORE), "performance.wav", "midi")
    with pytest.raises(ValueError, match="^Invalid score file: "):
        Matchmaker("missing.musicxml", str(PERFORMANCE), "midi")
    with pytest.raises(NotImplementedError):
        Matchmaker(str(SCORE), None, "midi")
    with pytest.raises(NotImplementedError):
        make(stream=object())
    with pytest.raises(NotImplementedError):
        make(device_name_or_index=0)


def test_kwargs_select_processor_and_stream():
    mm = make(kwargs={"processor": "chord_onset", "polling_period": 0.02})
    assert mm.polling_period == 0.02 and mm.config == {}
    assert len(list(mm.run(verbose=False))) > 0
    assert len(list(make(processor="chord_onset").run(verbose=False))) > 0

    # replacing the defaults without a polling period makes outerhmm event-based, like the reference
    event = make("outerhmm", kwargs={"piano_range": True})
    assert event.polling_period is None
    assert len(list(event.run(verbose=False))) > 0

    with pytest.raises(ValueError, match="allowed"):
        make(kwargs={"window_size": 3})
    with pytest.raises(ValueError, match="processor"):
        make("hmm", processor="chord_onset")
    with pytest.raises(ValueError):
        make(kwargs={"polling_period": -1.0})


def test_build_score_annotations():
    mm = make()
    notes = mm.build_score_annotations(level="note")
    np.testing.assert_array_equal(notes, mm.score_positions)
    beats = mm.build_score_annotations()
    assert beats.dtype == np.float32
    assert beats[0] == np.ceil(mm.score_positions.min()) and beats[-1] == np.floor(mm.score_positions.max())
    with pytest.raises(ValueError, match="Invalid score annotation level"):
        mm.build_score_annotations(level="bar")
    with pytest.raises(ValueError, match="Invalid return type"):
        mm.build_score_annotations(return_type="frames")
    with pytest.raises(NotImplementedError):
        mm.build_score_annotations(return_type="seconds")
    with pytest.raises(NotImplementedError):
        mm.build_score_annotations(musical_beat=True)
