# matchmaker-rs

[![CI](https://github.com/Meisterschueler/matchmaker-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/Meisterschueler/matchmaker-rs/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2FMeisterschueler%2Fmatchmaker-rs%2Fmaster%2FCargo.toml&query=%24.package.version&label=version)](Cargo.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Pure-Rust port of [pymatchmaker](https://github.com/pymatchmaker/matchmaker), a library for real-time music alignment (score following).

Status: work in progress, developed spec-driven with [Spec Kit](https://github.com/github/spec-kit) (see `specs/`).

## Reference implementation

- pymatchmaker release: tag [`v0.4.0`](https://github.com/pymatchmaker/matchmaker/tree/v0.4.0).
  Only released tags are used; see the constitution (`.specify/memory/constitution.md`, I).
- Python versions for the bindings follow its `requires-python` (currently 3.10–3.13).

## Roadmap

1. MIDI score following: methods `pthmm`, `hmm`, `outerhmm`; processors `pitch`, `chord_onset`.
2. Audio score following: methods `arzt`, `dixon`; processors `lse`, `chroma`.
3. Later: remaining methods (audio `outerhmm`, `skf`, `pfkorz`, `arzt_tempo`,
   `arzt_multi_fold`) and parangonar methods.

## Benchmark datasets

- (n)ASAP, Batik, Vienna4x22 (as in the paper), downloaded in CI.
- The bundled Mozart K265 example for fast tests.

## Paper

Park, Cancino-Chacón, Chiruthapudi, Nam: *Matchmaker: An Open-Source Library for Real-Time Piano Score Following and Systematic Evaluation*, ISMIR 2025. [arXiv:2510.10087](https://arxiv.org/abs/2510.10087)
