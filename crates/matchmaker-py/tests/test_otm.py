"""OTM defaults, parity, empty paths, and actionable input errors."""

import numpy as np
import pytest

from test_drop_in import GOLDEN, Matchmaker, PERFORMANCE, SCORE, make
from matchmaker_rs import AVAILABLE_METHODS, DEFAULT_KWARGS


def test_otm_defaults_and_empty_path():
    assert "OTM" in AVAILABLE_METHODS["midi"]
    assert "OTM" not in AVAILABLE_METHODS["audio"]
    assert DEFAULT_KWARGS["midi"]["OTM"] == {
        "processor": "parangonar",
        "polling_period": None,
    }
    mm = make("OTM")
    assert mm.polling_period is None
    assert mm.score_follower.alignment_path.shape == (2, 0)
    assert make().method == "pthmm"


def test_otm_matches_reference_with_explicit_tempo_configuration():
    mm = make("OTM", kwargs={"lookback": 1, "init_beat_period": 0.5})
    expected = GOLDEN / "simple_mozart_k265_var1" / "OTM"
    np.testing.assert_allclose(
        list(mm.run(verbose=False)), np.load(expected / "positions.npy"), rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        mm.score_follower.alignment_path,
        np.load(expected / "alignment_path.npy"),
        rtol=0, atol=1e-6,
    )


@pytest.mark.parametrize("kwargs,key", [
    ({"dim_model": 32}, "dim_model"),
    ({"polling_period": 0.01}, "polling_period"),
    ({"lookback": 0}, "lookback"),
    ({"unknown": 1}, "unknown"),
])
def test_otm_rejects_invalid_configuration(kwargs, key):
    with pytest.raises(ValueError, match=key):
        make("OTM", kwargs=kwargs)


def test_otm_is_midi_only_and_other_parangonar_methods_remain_deferred():
    with pytest.raises(ValueError, match="OTM"):
        Matchmaker(SCORE, PERFORMANCE, input_type="audio", method="OTM")
    for method in ["OPTM", "SLT_OLTW", "SL_OLTW"]:
        with pytest.raises(ValueError):
            make(method)
