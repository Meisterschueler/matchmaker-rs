# Data Model: Follow Upstream pymatchmaker 0.5.1

Entities this sync introduces or changes. Existing entities (Observation, AlignmentPath,
StepDebug, MethodConfig, Score) are extended, not redefined.

## Score compound events (`score_cevts`)

Port of the `score_cevts` reference builder (`matchmaker/registry.py`): the reference
features of the Dannenberg matcher.

| Field | Type | Meaning |
|---|---|---|
| `cevts` | `Vec<u128>` | One pitch-set bitmask per unique `onset_beat` of the score note array; bit *p* set ⇔ MIDI pitch *p* in the compound event |
| `score_positions` | `Vec<f32>` | The unique onset beats, ascending; same length as `cevts` |

Invariants:
- Built from the (optionally unfolded) score part's note array, like `onset_pianoroll`.
- `score_positions` is strictly increasing; every `cevts[i]` is non-empty (every onset
  has at least one note).
- `piano_range` does not apply: pitches are raw MIDI numbers 0–127 (upstream passes
  `piano_range: false` to the processor and full pitch sets to the follower).

## Dannenberg matcher state (`DPDannenberg` port)

| Field | Upstream | Type | Meaning |
|---|---|---|---|
| `values` | `_values` | `Vec<f64>` | Current DP rating column, one entry per compound event (row) |
| `used` | `_used` | `Vec<u128>` | Pitches already credited to each row in the current column |
| `prev_values`, `prev_used` | `_prev_values`, `_prev_used` | same | Previous column (shifted at each incoming note) |
| `best_value` | `_best_value` | `f64` | Best rating seen so far (monotone; starts −∞) |
| `best_row` / `current_index` | `_best_row` | `usize` | Row of the best rating; the reported score event |
| `prev_note_time` | `_prev_note_time` | `Option<f64>` | Time of the previous note (only feeds the unused cevt-grouping test) |
| `beats_per_second` | `_beats_per_second` | `f64` | Smoothed tempo, seeded `tempo / 60` |
| `anchor_beat`, `anchor_time` | `_anchor_beat`, `_anchor_time` | `f64` | Last matched beat and its performance time |
| `prev_match_time`, `prev_match_beat` | same | `Option<f64>` | Previous best-match update, for tempo smoothing |
| config: `c_m`, `c_e`, `epsilon`, `epsilon_fraction`, `window`, `smoothing` | same | `f64`/`usize` | See [contracts/sync-api.md](contracts/sync-api.md) |

State transitions per incoming note (one MIDI pitch, event-based stream):

1. Cevt-grouping test (`_is_new_cevt`): computed from `prev_note_time`, `epsilon`,
   `epsilon_fraction`, predicted next-event interval — **result discarded** (upstream dead
   branch, ported as written); `prev_note_time` updated.
2. Column shift: `prev_values/prev_used ← values/used` on rows `lo..hi`,
   `lo = max(0, best_row − 10)`, `hi = min(n_states, best_row + window)`.
3. DP update on `lo..hi`: predecessors same-row (extend, `−c_e` if pitch ∉ cevt; `+1` and
   credit pitch if pitch ∈ cevt and uncredited), diagonal (advance: `prev[r−1] + 1 −
   c_m·missing(r−1)`, credit set `{pitch}`), vertical (skip: `values[r−1] −
   c_m·missing(r−1)`, `≥` wins, credit set emptied).
4. Best update: only if the window argmax strictly exceeds `best_value`; then tempo
   smoothing (`smoothing·old + (1−smoothing)·rate`) gated by `elapsed > 0.05 s` and
   `0 < advanced < 16` beats; anchor moved.

Invariants:
- Rows outside `lo..hi` keep stale values (upstream behavior; port as written).
- Initial column: `values[0] = 0`, `values[r] = −c_m · Σ|cevts[0..r]|` — matching must
  start at the first compound event.
- `best_value` and `current_index` never regress within a run; `reset` restores the
  initial column and anchors.
- Reported position = `anchor_beat + beats_per_second · (t − anchor_time)`, capped at
  `score_positions[current_index + 1]`; following stops past the last row.

## Smoothness window (SPARC, revised)

| Field | Type | Meaning |
|---|---|---|
| `t0`, `t1` | `f64` | Onset-anchored window bounds (unchanged: 30 s forward windows) |
| sample range | `(usize, usize)` | `searchsorted` indices into the 3000 Hz resampled held-position path; windows sharing a range share one value |
| value | `Option<f64>` | Spectral arc length of `|gradient(position)|`, or `None` |
| reason (reference) | enum | `ok`, `stationary`, `undefined_band`, `uncovered`, `insufficient_samples` — the Rust port keeps only the inclusion/exclusion semantics and returns the mean |

Invariants:
- Resampling rate `SPARC_FS = 3000.0` (was 50.0); grid is `np.arange(first t0, last t1,
  1/fs)` with NumPy fill semantics; position holds the last report, NaN before the first.
- Windows not fully covered by reports, or with `< 2` samples, are excluded from the mean.
- The sliding-spectrum route and the per-window FFT route produce the same values; the
  cost heuristic only selects between them. Edge samples of each window are corrected for
  one-sided gradients.

## Pin metadata

| Field | Where | 0.4.0 → 0.5.1 |
|---|---|---|
| `pymatchmaker` pin | `tools/golden/pyproject.toml`, `uv.lock` | git tag `v0.5.1` |
| `REFERENCE_TAG` | `tools/golden/generate.py`, `generate_audio.py` | `"v0.5.1"` (embedded in every golden `meta.json`) |
| Version table | `UPSTREAM.md` | pymatchmaker 0.5.1; partitura/hiddenmarkov rows follow the regenerated lockfile |
| Reference tag link | `README.md` | `v0.5.1` |
