# Contract: Python API (`matchmaker_rs`)

Built from `crates/matchmaker-py` with maturin (`abi3`, CPython 3.10–3.13, following the
reference's `requires-python`). Drop-in for pymatchmaker's MIDI usage (FR-014, US5).

## Entry point

```python
from matchmaker_rs import Matchmaker   # instead of: from matchmaker import Matchmaker

mm = Matchmaker(
    score_file="simple_mozart_k265_var1.musicxml",
    performance_file="simple_mozart_k265_var1.mid",
    input_type="midi",
    method="pthmm",            # "pthmm" (default) | "hmm" | "outerhmm"
    processor=None,            # "pitch" | "chord_onset" | None (method default)
    tempo=None,
    wait=False,                # accepted; simulation runs as fast as possible either way
    unfold_score=True,
    kwargs=None,               # same keys as the reference: processor, piano_range, polling_period, method params
)
for beat in mm.run(verbose=False):
    print(beat)                # float, score beats
path = mm.score_follower.alignment_path   # numpy.ndarray, shape (2, T), float64
```

Same signature as the reference `Matchmaker.__init__` (positional order, keyword-only after
`method`), including `stream` and `device_name_or_index` (accepted only as `None`; anything
else raises `NotImplementedError` naming the supported options).

## Attributes and methods (reference shapes)

| Name | Type | Notes |
|---|---|---|
| `run(verbose=True)` | generator of `float` | generator return value = alignment path, like the reference |
| `score_follower.alignment_path` | `ndarray (2, T)` | row 0 perf seconds, row 1 score beats |
| `score_follower._alignment_path` | `list[tuple[float, float]]` | for code that reads the private list |
| `score_follower.score_positions` | `ndarray float32` | follower states (incl. inserted states for `pthmm`/`hmm`) |
| `mm.score_positions` | `ndarray float32` | unique score onsets (`np.unique(onset_beat)`), as in the reference |
| `tempo`, `method`, `input_type`, `config`, `polling_period` | as reference | |
| `build_score_annotations(level="beat", musical_beat=False, return_type="beats")` | `ndarray` | |
| `export(directory)` | `None` | extra: Rust export (FR-018) |

## Errors

| Situation | Exception |
|---|---|
| `input_type="audio"` or unknown method | `ValueError`, message lists available methods (US5-2) |
| invalid score file | `ValueError("Invalid score file: …")` like the reference |
| performance file not `.mid`/`.midi` | `ValueError` with the reference's message |
| other library errors | `ValueError` / `OSError` carrying the Rust error text |

## Module-level

`AVAILABLE_METHODS = {"midi": ["pthmm", "hmm", "outerhmm"]}`, `DEFAULT_KWARGS` (MIDI part),
`__version__`, `__reference_version__` (reference tag the build was validated against).
