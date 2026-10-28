<!--
Sync Impact Report
- Version: none → 1.0.0 (initial ratification)
- Principles: I. Reference Parity (NON-NEGOTIABLE), II. Readability & Structural Parity,
  III. Visible Performance, IV. Debuggability, V. Pure Rust, Libraries First
- Sections: Conventions, Core Principles, Scope & Technical Constraints,
  Development Workflow & Quality Gates, Governance
- Dependent files: plan-template.md ✅ (generic Constitution Check), spec-template.md ✅,
  tasks-template.md ✅, .github/pull_request_template.md ✅, deny.toml ✅, README.md ✅,
  specs/001-midi-score-following/spec.md ✅
- Follow-up: none
-->
# matchmaker-rs Constitution

## Conventions

- The key words MUST, MUST NOT, SHOULD, SHOULD NOT and MAY are to be interpreted as
  described in RFC 2119.
- Facts that change during normal development live in `README.md`, not here: the pinned
  reference release, the delivery roadmap, and the benchmark datasets. Changing them MUST NOT
  require an amendment, but MUST follow the principles below (e.g. I for the reference).
- Each rule is stated once; other sections refer to it instead of repeating it.

## Core Principles

Principles are numbered by precedence: when two conflict, the lower-numbered one wins.
I is non-negotiable and MUST NOT be traded off. Any other deviation MUST be justified in the
plan's complexity tracking, naming the conflicting principles.

Example: if a recommended crate (V) cannot meet the golden-test tolerance (I), own code is
written and the deviation from V is justified.

### I. Reference Parity (NON-NEGOTIABLE)

- The reference implementation MUST be a released pymatchmaker version (tag recorded in
  `README.md`). Unreleased states (e.g. the `develop` branch or commits between releases)
  MUST NOT be used. Moving to a newer release MAY happen at any time, but MUST be an
  explicit, reviewed change that regenerates the golden data.
- The Python environment that generates golden data MUST be locked (lockfile checked in next
  to the generator scripts). Versions of transitive dependencies (e.g. partitura, numpy) are
  defined by this lockfile, because the reference only declares lower bounds.
- Every processor and every alignment method MUST have golden tests: a checked-in Python
  script generates reference outputs (features, positions, alignment path) from fixed inputs;
  Rust MUST match them within a tolerance documented next to each test.
- Where bit-identity is impossible, parity MUST be shown via the evaluation metrics
  (accuracy, error in beats) instead, and the deviation MUST be documented. This applies to:
  - score synthesis (rustysynth vs FluidSynth);
  - stochastic methods, whose random streams cannot match numpy's.
- Evaluation metrics and the alignment mapping function MUST follow the reference code
  (`evaluate_alignment()`) exactly. Where the reference code and the paper (ISMIR 2025, §5.2)
  disagree, the code at the reference tag MUST be followed and the difference documented.

Rationale: a port is only trustworthy if every result can be compared to the original.

### II. Readability & Structural Parity

- The crate MUST mirror pymatchmaker's pipeline: `Stream` → `Processor` → `OnlineAlignment`.
- Each alignment method and feature processor MUST live in its own module whose name maps
  to the Python method/processor key (e.g. `pthmm`, `chord_onset`).
- Every module that ports Python code MUST name the ported Python path in its module doc
  (e.g. `matchmaker.prob.hmm.PitchHMM`) and, where applicable, the paper section, but MUST
  NOT repeat the reference release (see Conventions).
- Every public item MUST be documented (lint `missing_docs`); every public module MUST
  contain at least one runnable doc example.
- `unsafe` MUST NOT be used unless a benchmark proves a measurable gain; such a block MUST
  carry a `// SAFETY:` comment stating its invariant (lints `unsafe_code`,
  `clippy::undocumented_unsafe_blocks`).

Rationale: other people must be able to read the Rust code side by side with the Python
code and the papers.

### III. Visible Performance

- Every pipeline stage (stream, feature extraction, inference) MUST be instrumented with
  `tracing` spans so per-frame latency is measurable without code changes. Per-frame spans
  MUST use level `DEBUG` or lower; benchmarks MUST run without an active subscriber so that
  instrumentation does not distort the measured latency.
- A benchmark harness MUST run Rust and Python on identical inputs and report, side by side:
  feature latency (ms/frame), inference latency (ms/frame), real-time factor, and the
  accuracy metrics of the reference.
- CI MUST run the full benchmark on every push to `master` and on every PR that touches
  algorithm code, and MUST publish the report; the README MUST link to the latest report.
- Latency is compared on one reference platform (CI `ubuntu-latest`) using the median of at
  least 5 runs. A slowdown of more than 10 % against the latest `master` report is a
  regression; the report MUST flag it and the PR MUST justify it.

Rationale: performance is the reason for the port; it must be measured, never assumed.

### IV. Debuggability

- Intermediate results (features, reference features, cost/probability windows, alignment
  path) MUST be exportable in a format Python reads directly (`.npy` or CSV), via the
  library and a CLI flag.
- Log verbosity MUST be controllable via `RUST_LOG`.
- Stochastic methods MUST accept a seed; with a fixed seed, runs MUST be reproducible.
- Library errors MUST be typed (e.g. via `thiserror`) and state what failed and with which
  input; the CLI MAY wrap them (e.g. via `anyhow`). Invalid user input MUST NOT cause a
  panic.

Rationale: when Rust and Python disagree, the first differing stage must be findable fast.

### V. Pure Rust, Libraries First

- The default build MUST NOT link C/C++ code (no C-compiling `build.rs`, no `-sys` crates
  wrapping C libraries) and MUST build for `wasm32-unknown-unknown`.
- Only the following MAY link C/C++ or platform code, and only outside the default build
  (optional, non-default cargo feature or separate crate):
  - the PyO3 bindings (link against libpython);
  - live device I/O, if no pure-Rust option exists.
- Before writing new code, an existing, maintained crate MUST be evaluated and SHOULD be
  used (e.g. `midly` for MIDI, `musicxml` for MusicXML, `rustfft`/`realfft` for FFT,
  `rustysynth` for synthesis).
- Every new dependency MUST be license-compatible with Apache-2.0 (checked by `cargo deny`)
  and justified in the plan.

Rationale: easy builds on every platform including WebAssembly; less own code means less to
maintain.

## Scope & Technical Constraints

- Language: Rust stable, edition 2024; the MSRV is declared as `rust-version` in
  `Cargo.toml`. Documentation, specs, and code comments MUST be in English.
- Interfaces: library crate, CLI binary, and PyO3 bindings. For every delivered method and
  processor, the bindings MUST be a drop-in for the corresponding pymatchmaker API; supported
  Python versions follow the reference's `requires-python`.
- Score input: MusicXML and MIDI. Score features (note array, `onset_beat`, repeat
  unfolding) MUST match partitura's output (version per I's lockfile) for the reference
  inputs.
- License: Apache-2.0; `NOTICE` attributes pymatchmaker.

## Development Workflow & Quality Gates

- Work follows Spec Kit: specify → (clarify) → plan → tasks → (analyze) → implement.
- Default branch: `master`. Changes land via pull requests with green CI.
- CI (Linux, Windows, macOS) MUST pass: `cargo fmt --check`, `cargo clippy --all-targets
  --all-features` with warnings as errors, and `cargo test --all-features` including the
  golden tests (I).
- CI MUST additionally pass: a build for `wasm32-unknown-unknown` (V), a check with the
  MSRV toolchain, and `cargo deny check` (V). The benchmark runs as defined in III.
- Every PR touching an algorithm MUST include or update its golden tests (I) and link the
  benchmark delta (III).

## Governance

- This constitution supersedes other practices in this repository.
- Amendments MUST be made via a PR that updates this file and its Sync Impact Report, states
  the version bump with rationale, and adjusts affected specs/plans. The repository
  maintainer approves amendments.
- Versioning follows semantic versioning: MAJOR for removed or redefined principles, MINOR
  for new principles, sections, or rules, PATCH for clarifications.
- Compliance is checked at two points: every plan MUST contain a Constitution Check (any
  violation MUST be justified in the plan's complexity tracking), and every PR MUST complete
  the checklist in `.github/pull_request_template.md`.

**Version**: 1.0.0 | **Ratified**: 2026-10-28 | **Last Amended**: 2026-10-28
