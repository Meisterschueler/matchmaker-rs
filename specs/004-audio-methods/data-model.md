# Data Model: Remaining Audio Methods and Features

## Audio configuration and feature observations

Extend AudioMethod and AudioProcessorKind one story at a time. Preserve existing public field meanings and constructors. Method-specific options validate at construction; no future variants or broad config rewrite. Rates/hops/transform sizes must be positive, method scalar parameters finite, particle counts positive, and seed an unsigned integer. Reference infinity normalization is a named norm choice, not an accidental non-finite numeric parameter.

Vector processors retain Observation { features: Vec<f32>, perf_time: f64 }. Returned dimensions and batch-to-stream timestamps follow the reference adapter, including trimmed CQT rows. "Feature dimensions must match the selected processor and reference model." A short input may buffer/produce no observation exactly as the reference; no extra timestamp shifts.

KorzeniowskiObservation is separate: spectrum, onset, loudness, and optional active_notes/onset_notes. Audio sets both optional note fields absent. Spectrum uses reference numeric precision; onset is nonnegative but not artificially capped. Export these fields separately. "Structured Korzeniowski observations must not be fed to vector-only followers."

## Reference models

| Entity | Fields / relationships | Validation |
|---|---|---|
| ScoreAudioReference | Rendered score waveform (peak-normalized), feature frames from one whole-signal pass, frame-to-beat mapping | Same processor configuration as performance; frame `j` starts at `j*hop`; a score shorter than one window is an `AudioConfig` error; see research R11 for the level and framing conventions and how to replace them |
| MultiFoldHypothesis | Score unfolding/path, feature reference, beat domain, follower state, recent costs | Valid repeat path; selection history bounded by selection_window |
| SkfScoreModel | Chord/onset sequence, normalized spectrum templates, synthesis/fallback identity | Template width n_fft/2+1; covariance/likelihood safeguards match reference; the mode (`synthesized` or `gaussian`) and the SoundFont path travel with the model into every export |
| AudioOuterHmmScoreModel | Weighted chords, onset beats, 88-bin pitch/chord templates, sound/pause transitions | Learned-overlay mask and template rate/hop match recorded provenance |
| KorzeniowskiScoreModel | beat_grid, onset_positions, nearest_onsets, rest_mask, templates, active_notes | Consistent grid lengths, template width n_fft/2+1, valid nearest-onset indices |

Korzeniowski score defaults: beat_resolution=0.1, num_harmonics=10, harmonic_bandwidth=15 Hz. Preserve onset_beat versus onset_quarter/time-signature semantics in every model. "Asset hashes, provenance, and template mode must accompany every template comparison."

## Follower state and lifecycle

- ArztTempoState: Current tempo, insertion/deletion bookkeeping and historical costs needed by ArztTempoModel; reset with follower.
- MultiFoldState: Candidate follower states, current hypothesis, selection history, switch diagnostics.
- SkfState: Active chord/age hypotheses, normalized weights, conditional tempo moments, beam/merge diagnostics.
- AudioOuterHmmState: Event index, sound/pause posterior vector, transition/likelihood diagnostics.
- PfkorzState: Seeded RNG, position/tempo particles, normalized weights, resampling/rest/onset diagnostics.

Lifecycle: validate input/config -> build reference -> reset processor -> consume timestamped observations -> emit position and diagnostic state -> finalize/export/evaluate. Repeated runs build/reset state without prior-run contamination. Degenerate-state handling follows the reference and yields typed failures where external data violates invariants.

## Evaluation and delivery evidence

GroundTruth reuses the existing evaluator model; audio accepts direct perf_sec/score_beat TSV. Note-ID match formats need an explicitly supplied MIDI performance, never the audio file passed into a MIDI loader.

EvaluationRecord includes reference mapping/metrics, fixture identity, seed, template mode/hash, processor settings, and benchmark repetitions. An alignment path retains existing performance seconds / score beats semantics, shape, and intentional history allocation.

DeliveryEvidence records C01-C12 identity, associated tasks, golden maxima/metrics, benchmark delta, platform gates, and eventual commit hash. Planning does not populate runtime results. "A story is complete only after its own checks pass and its separately authorized commit exists."
