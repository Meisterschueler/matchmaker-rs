# Research: Online Transformer Matcher (OTM)

**Date**: 2026-11-24

## R1. Reference release and dependency baseline

- **Decision**: Use only the repository-pinned pymatchmaker `v0.5.1` and its checked-in
  `tools/golden/uv.lock`; its resolved Parangonar version is `3.4.1`.
- **Rationale**: Principle I requires a released, locked reference. The release's
  `matchmaker/methods.yaml` registers `OTM` for MIDI with the external Parangonar adapter,
  the `parangonar` processor, event-based input and `polling_period: null`. The lock includes
  `parangonar[accelerated]` and PyTorch for reference generation; neither is a Rust runtime
  dependency.
- **Alternatives considered**: Tracking pymatchmaker `main` or an unpinned Parangonar version
  would make generated reference results move without an explicit reviewed re-pin.
- **Evidence**:
  - [pymatchmaker v0.5.1 `methods.yaml`](https://github.com/pymatchmaker/matchmaker/blob/v0.5.1/matchmaker/methods.yaml)
  - [pymatchmaker v0.5.1 adapter](https://github.com/pymatchmaker/matchmaker/blob/v0.5.1/matchmaker/external/__init__.py)
  - `tools/golden/uv.lock`: pymatchmaker `0.5.1`, Parangonar `3.4.1`, PyTorch `2.14.1`.

## R2. Adapter behavior and online semantics

- **Decision**: Port the behavior reached through `OnlineParangonarAlignment.__init__`,
  `step`, and `OnlineTransformerMatcher.__call__`; do not port the offline alignment entry
  point, `online_legacy`, or `OnlinePureTransformerMatcher`.
- **Rationale**: The pymatchmaker adapter constructs `OnlineTransformerMatcher` from the
  score note array, invokes it once per processed performance note, clamps the returned onset
  index, and maps it through the unique score onsets. The `parangonar` processor emits one
  positive-velocity note-on at a time and rejects a frame with more than one such event.
  The reference builder is `note_array_with_grace`; score IDs are made unique if absent or
  duplicated.
- **Behavior to preserve**:
  - Start from the first performed note and an initial beat period of `0.5` seconds per
    score beat; the pymatchmaker adapter does not replace this with the score's marked tempo.
  - Build tokens from the last eight performance pitches and up to sixteen score-onset pitch
    sets around the current index, then pad the combined sequence to 26 positions.
  - Use the tracker's greedy current-onset match, transformer prediction, prediction-window
    checks, tempo-aware jump decisions, and stuck-input recovery in the same order as
    `__call__`.
  - A score pitch set excludes grace notes, while the full note list used for pitch matching
    includes them. The adapter exposes the unique onset index as the score position and stops
    when it reaches the final score onset.
- **Alternatives considered**: A generic transformer-only tracker would omit the heuristic
  and tempo-model behavior that makes `OTM` distinct from `OPTM`.
- **Evidence**:
  - [pymatchmaker adapter](https://github.com/pymatchmaker/matchmaker/blob/v0.5.1/matchmaker/external/__init__.py)
  - [Parangonar 3.4.1 `online_matchers.py`](https://github.com/sildater/parangonar/blob/3.4.1/parangonar/match/online_matchers.py)

## R3. Model architecture and weights

- **Decision**: Implement the fixed inference graph used by OTM: 91-entry, 64-dimensional
  embedding; seven token attributes summed per position; sinusoidal position encoding;
  six post-normalized transformer encoder layers with eight heads and a 64-wide ReLU
  feed-forward layer; and a two-class output projection. Inference uses evaluation mode, so
  dropout is disabled.
- **Rationale**: This is the architecture selected by the OTM constructor and checkpoint,
  not the different defaults on the reusable `AlignmentTransformer` class. Pymatchmaker
  obtains a class-1 decision by softmaxing over sequence positions and selecting the maximum.
- **Weight artifact**: Parangonar 3.4.1 distributes
  `assets/alignment_transformer_checkpoint.pt` (13,460,509 bytes; SHA-256
  `9ec8a1b8e6731c01a9bb359635dab3fa0b9e60045bbbf18f27227237b249af97`). Inspection of its
  archive metadata found a model state dict with 76 float32 tensors, plus small optimizer
  metadata and no optimizer tensor state. A 50,000-position encoding buffer accounts for
  12,800,000 bytes; the OTM token sequence is always 26 positions. The learned parameters
  are about 157,000 float32 values.
- **Decision on conversion**: Convert the state dict offline in the locked Python reference
  environment into a versioned, checksummed compact Rust-readable artifact. Keep the 26
  position-encoding rows used by OTM so their values are identical to the released
  checkpoint; omit the unused 49,974 rows and optimizer metadata. Never deserialize the
  Python pickle from the Rust runtime.
- **Alternatives considered**: Shipping the complete `.pt` file would retain a large
  deterministic buffer unused by the fixed input shape and would require unsafe/untrusted
  pickle handling; regenerating positional rows at runtime risks small numeric differences.
- **Evidence**:
  - [Parangonar 3.4.1 `pretrained_models.py`](https://github.com/sildater/parangonar/blob/3.4.1/parangonar/match/pretrained_models.py)
  - [Parangonar 3.4.1 `online_matchers.py`](https://github.com/sildater/parangonar/blob/3.4.1/parangonar/match/online_matchers.py)
  - [Parangonar package assets](https://github.com/sildater/parangonar/tree/3.4.1/parangonar/assets)
  - [Parangonar 3.4.1 PyPI wheel](https://pypi.org/project/parangonar/3.4.1/)

## R4. Inference implementation and platform constraints

- **Decision**: Implement the fixed OTM inference operations in Rust with no new default
  dependency. Reuse existing `libm` for scalar functions and existing serialization/runtime
  facilities. Load a validated, compact model artifact as data.
- **Rationale**: The model is a small fixed graph with short sequences and 64-wide layers.
  This avoids adding a large ML runtime to the default build and preserves the repository's
  no-C/C++ and `wasm32-unknown-unknown` requirements.
- **Alternatives considered**:
  - Candle and Burn offer Rust model operations but add substantial dependency trees for
    this fixed inference graph and complicate default-WASM/runtime validation.
  - `tract` supports exported ONNX models but adds a larger runtime and its current MSRV
    exceeds this repository's MSRV 1.88.
  - ONNX Runtime would add a native C/C++ runtime and conflicts with Principle V.
- **Required verification**: Compare tokenizer output, each transformer layer's outputs,
  final logits, selected onset index, tempo state, positions, and alignment path against
  golden data. Benchmark the native pass before optimizing; do not rely on an unverified
  assumption that a handwritten matrix loop is faster.
- **Evidence**:
  - [Candle](https://github.com/huggingface/candle)
  - [Burn](https://github.com/tracel-ai/burn)
  - [tract](https://github.com/sonos/tract)
  - Repository constraints in `.specify/memory/constitution.md` and `Cargo.toml`.

## R5. Model asset licensing and provenance

- **Decision**: Distribute the converted checkpoint under Apache-2.0 following the explicit
  project-owner determination below. Record the source, checksum, license and notices; do
  not describe the owner's determination as independent upstream/legal confirmation.
- **Rationale**: Parangonar 3.4.1 is packaged under Apache-2.0 and includes the checkpoint,
  but the wheel/repository provides no checkpoint-specific license, model card, or training
  data provenance. Package-level licensing is useful evidence but does not independently
  establish the weights' provenance or training-data terms.
- **Alternatives considered**: Treating the package license as conclusive and bundling
  immediately would make an unsupported assumption about a trained artifact. Requiring a
  Python/PyTorch runtime or a network download at inference would conflict with the feature
  requirements.
- **Original implementation gate**: Obtain maintainer confirmation or another documented
  project decision before packaging. The owner supplied that decision during implementation.
  If the project later determines the terms do not permit redistribution, stop distribution
  and revise the requirement; do not silently substitute weights.
- **Evidence**:
  - Parangonar 3.4.1 wheel metadata and bundled `LICENSE` declare Apache-2.0.
  - The checkpoint has no separate license file or model-card entry in the wheel.
  - The project's [model asset](https://github.com/sildater/parangonar/blob/3.4.1/parangonar/assets/alignment_transformer_checkpoint.pt)
    is bundled upstream; its training-data provenance is not documented in the inspected
    package.

### Implementation preflight and project decision (2026-11-24)

- **Status**: T001 is cleared by an explicit project-owner decision. The owner states
  that Parangonar, including its weights, is Apache-2.0 and authorizes porting and
  redistribution under that license. This records the project's license determination;
  it is not an independently obtained statement from the upstream maintainers.
- Rechecked the [release README](https://github.com/sildater/parangonar/blob/3.4.1/README.md).
  Its license section explicitly says, "The code in this package is licensed under the
  Apache 2.0 License." It does not explicitly address the trained weights.
- Rechecked the [release asset listing](https://api.github.com/repos/sildater/parangonar/contents/parangonar/assets?ref=3.4.1).
  It includes the checkpoint but no adjacent model license or model card.
- A GitHub issue/PR search in `sildater/parangonar` for `license`, `licensing`,
  `redistribution`, or `provenance` returned no results. This limited search does not
  establish that no permission exists elsewhere.
- The project owner resolved the gate after these checks: "parangonar ist komplett und
  definitiv unter der apache-2.0-lizenz. d.h. du darfst alles, sowohl source-code, als auch
  gewichte, unter der apache-2.0-lizenz nach matchmaker-rs portieren."
- Proceed using Apache-2.0 for the source and converted weights, retaining upstream
  attribution, license, source checkpoint hash, and conversion metadata. Training-data
  provenance remains undocumented in the inspected upstream package.

### Verified implementation findings

- The compact artifact is 635,656 bytes; SHA-256
  `dbac0983fda23a46bc44aa20abe896d73d7ba87013094fa7b0c5da69e6d397d9`.
  Re-running the offline converter preserves the binary checksum.
- Upstream `__call__` explicitly initializes the tempo model at `0.5`, ignoring its stored
  `init_beat_period`. The accepted parameter remains validated; the behavior is documented
  rather than silently changed.
- The greedy branch lacks the model branch's grace guard. Preserve its tempo update when
  a grace candidate shares a pitch with the current non-grace context.
- NumPy 2 uses float32 for tempo initialization, repeated-onset medians, and interpolation.
  SciPy uses weighted-endpoint interpolation, not the algebraically equivalent slope
  expression. Mirroring float64 instead produced a measurable 1.8e-6 tempo-state divergence
  on Mozart; matching the reference arithmetic resolves it.
- Compare sequence-softmax probabilities before argmax. Distinct near-zero float32 logits
  can round to equal probabilities, in which case PyTorch chooses the first token.
- All six OTM fixtures pass score/token/model/tempo/index/path comparisons, including a
  dedicated grace fixture. Native/CLI/Python paths are identical to the reference.

## R6. Score representation and OTM-specific data

- **Decision**: Add an OTM score-reference builder that preserves the reference's note IDs,
  pitches, beat onsets, grace-note flags, and distinct onset positions. Extend the existing
  score-note representation to retain `is_grace`; do not infer grace status from duration.
- **Rationale**: The current Rust `ScoreNote` has no grace flag, while OTM uses
  `note_array_with_grace` and treats grace notes differently in its score pitch sets and
  note matching. The reference adapter replaces all IDs if any are empty or duplicated.
- **Alternatives considered**: Inferring grace notes from zero duration loses the explicit
  Partitura class distinction and can misclassify other zero-duration notes. Reusing the
  onset pianoroll would discard note-level IDs and grace semantics.
- **Evidence**:
  - `src/score/mod.rs` `ScoreNote` and `note_array` currently omit `is_grace`.
  - `src/score/musicxml.rs` distinguishes `NoteClass::GraceNote` during parsing.
  - [pymatchmaker v0.5.1 adapter](https://github.com/pymatchmaker/matchmaker/blob/v0.5.1/matchmaker/external/__init__.py)

## R7. Golden tests, exports and benchmarks

- **Decision**: Extend the existing golden generator, comparator-compatible export layout,
  and benchmark method lists with OTM. Record per-call token inputs, logits when inference
  runs, selected positions and the tempo diagnostics needed to locate the first divergence.
- **Rationale**: The score and event processor are part of parity; comparing only final beats
  would make transformer or tokenization regressions difficult to diagnose. The reference
  requires its locked accelerated extra, already present in `tools/golden/uv.lock`.
- **Alternatives considered**: Testing only final alignment metrics would not identify
  whether differences originate in score construction, tokenization, model math, or
  post-processing.
- **Evidence**:
  - `tools/golden/generate.py` currently covers the six shipped MIDI methods.
  - `tests/common/mod.rs`, `src/export.rs`, and `tools/bench/` provide the comparison and
    export patterns to extend.
  - `.specify/memory/constitution.md`, Principles I, III and IV.
