# Contract: Golden Data and Export Layout

One layout for both the reference outputs (`tests/golden/`, written by
`tools/golden/generate.py`) and the Rust export (`--export <DIR>`), so
`tools/golden/compare.py <golden_dir> <export_dir>` can diff them stage by stage (US6, SC-007).
All arrays are `.npy` (NumPy format 1.0, little-endian), loadable with `numpy.load`; they contain
numeric dtypes only (strings go to JSON), so they are easy to read from Rust.

## Directory layout

```text
<root>/<piece>/
├── score/
│   ├── note_array.npy            structured: onset_beat <f4, duration_beat <f4, onset_quarter <f4,
│   │                             duration_quarter <f4, onset_div <i4, duration_div <i4, pitch <i4, voice <i4
│   ├── note_ids.json             ["n1-1", …] note ids in note_array order
│   └── meta.json                 {"score_tempo": 90.0 | null, "tempo": 90.0, "unfolded": true, "features": {…}}
├── stream/<mode>/                mode = event | windowed_0.01 | windowed_0.001
│   ├── messages.npy              structured: kind <i4 (0 note_on, 1 note_off, 2 control_change, 3 program_change),
│   │                             channel, note, velocity, control, value, program <i4 (-1 if n/a), time <f8;
│   │                             windowed: messages in frame order
│   ├── frame_offsets.npy         i8, (n_frames + 1,) windowed only: frame k = messages[offsets[k]:offsets[k+1]]
│   └── frame_times.npy           f8, (n_frames,) windowed only
├── processor/<name>/             pitch_piano_vector, pitch_piano_list, pitch_full_vector (windowed 0.01),
│   │                             pitch_piano_vector_event (event), pitch_piano_vector_0.001 (windowed 0.001,
│   │                             `outerhmm`), chord_onset_piano (windowed 0.001, incl. flush);
│   │                             a Rust export with another window gets the suffix `_<polling_period>`
│   ├── features.npy              f4, (T, 88|128) for vectors; pitch lists padded with -1, (T, max_len)
│   └── times.npy                 f8, (T,)
└── <method>/                     pthmm | hmm | outerhmm
    ├── score_positions.npy       f4, (N,)
    ├── observations_features.npy f4, input actually fed to the follower (copy of the processor output used)
    ├── observations_times.npy    f8
    ├── positions.npy             f8, (K,) every yielded position, in order
    ├── current_index.npy         i8, (K,)
    ├── probabilities.npy         f8, (K, N) forward variable / state distribution (omitted if K·N > 5e6)
    ├── beat_period.npy           f8, (K,)   hmm only
    ├── is_continuation.npy       bool, (K,) outerhmm only
    ├── alignment_path.npy        f8, (2, T)
    ├── evaluation.json           EvaluationResult keys of the reference (Rust: `follow --export D --ground-truth GT`)
    └── meta.json                 see below
```

`meta.json` (method level):

```json
{
  "reference": {"package": "pymatchmaker", "tag": "v0.4.0", "lockfile_sha256": "…"},
  "generator": "tools/golden/generate.py",
  "inputs": {"score": "simple_mozart_k265_var1.musicxml", "performance": "simple_mozart_k265_var1.mid",
             "ground_truth": "simple_mozart_k265_var1.match"},
  "config": {"method": "pthmm", "processor": "pitch", "piano_range": true, "polling_period": 0.01},
  "tolerances": {"positions": 1e-6, "probabilities_rtol": 1e-12, "f32_rtol": 1e-6, "metrics_abs": 1e-4}
}
```

For Rust exports `reference` is replaced by `{"implementation": "matchmaker-rs", "version": "…"}`.

## Pieces

`simple_mozart_k265_var1`, `mozart_k265_var1`, `Bach-fugue_bwv_858` (MusicXML + MIDI + `.match`
from the reference's `matchmaker/assets`, copied to `tests/data/`). Variants:

- `simple_mozart_k265_var1_midiscore` — the performance MIDI used as score (US1-2), all methods.
- `simple_mozart_k265_var1_wrongnote` — one note-on shifted by a semitone (US4-2), all methods.
- `fixture_<feature>` — minimal MusicXML scores for score features no bundled piece
  contains (repeats, voltas, grace notes, …); `score/` files only.

## Tolerances (documented next to each test, Principle I)

| Data | Tolerance | Basis |
|---|---|---|
| `note_array` ints/ids | exact | |
| `onset_beat`, `duration_*` | exact `f32` | same `f64` formula then cast |
| messages, frame times | exact | same formulas (R5); order compared per timestamp as multiset |
| processor features | exact | binary / integer values |
| `positions`, `alignment_path` | `< 1e-6` (SC-001); `hmm`: ≥ 99.9 % of steps (SC-002) | |
| `probabilities` | `rtol 1e-12` | summation order (R4) |
| `beat_period` | `rtol 1e-9` | mixed `f32`/`f64` (R4) |
| `evaluation.json` | `abs 1e-4`; `hmm` `abs 0.01` (SC-002) | 4-decimal rounding, FFT |

A failing comparison reports: method, piece, file, index of the first differing element,
expected and actual value (SC-007).

## Deviation classes

Every comparison in `tests/common/mod.rs::assert_close` prints one `DEVIATION` line (visible with
`cargo test -- --nocapture`): method, piece, file, class, max absolute deviation, max relative
deviation (elements `>= 1e-200`; below that only the absolute deviation is meaningful), length.

| Class | Meaning |
|---|---|
| `EXACT` | bit-equal to the golden data |
| `WITHIN_TOL` | differs, but inside the tolerance above (numpy-vs-non-numpy summation order: numpy sums pairwise with BLAS, the port sequentially) |
| `EXCEEDS` | outside the tolerance; the test fails |

Measured maxima (release build, all pieces and variants; `probabilities.npy` is compared per step):

| Method | File | Class | max abs | max rel |
|---|---|---|---|---|
| `pthmm` | `probabilities.npy` | `WITHIN_TOL` | 7.8e-16 | 1.8e-13 |
| `hmm` | `probabilities.npy` | `WITHIN_TOL` | 8.9e-16 | 9.3e-14 |
| `outerhmm` | `probabilities.npy` | `WITHIN_TOL` (6 of 1261 steps `EXACT`) | 1.8e-14 | 1.7e-13 |
| all | `positions`, `alignment_path`, `current_index`, `beat_period`, scores, stream, processors | `EXACT` | 0 | 0 |

The tolerance `rtol 1e-12` leaves about a factor of 6 headroom over the largest measured relative
deviation (1.8e-13). The benchmark report classifies the run-0 alignment path per piece and method
the same way (`EXACT`, `WITHIN_TOL`, `TIE`, `SHARE` for `hmm`, `EXCEEDS`) with the maximum position difference.

## Regeneration

```text
cd tools/golden
python -m uv sync --locked
python -m uv run generate.py --data ../../tests/data --out ../../tests/golden
```

Moving to a newer reference release = bump the tag in `pyproject.toml`, `uv lock`, regenerate,
review the diff (constitution I).
