# Research: Synchronize pymatchmaker 0.5.2

**Date**: 2026-11-27

**Upstream comparison**: [`v0.5.1...v0.5.2`](https://github.com/pymatchmaker/matchmaker/compare/v0.5.1...v0.5.2);
release tag `v0.5.2` points to commit
[`6e6f2713b5f6dc5dec51e939993b9c163189f3cf`](https://github.com/pymatchmaker/matchmaker/commit/6e6f2713b5f6dc5dec51e939993b9c163189f3cf).

The release diff changes four files:

| Upstream file | Change | Port decision |
|---|---|---|
| `CHANGES.md` | Release note: require python-hiddenmarkov 0.3.0 for the optimized forward step. | Record the change and its Rust mapping in this research and `UPSTREAM.md`. |
| `matchmaker/__init__.py` | Fallback package version changes from 0.5.1 to 0.5.2. | Update the active release pin and generated provenance; no Rust runtime version constant exists. |
| `matchmaker/prob/hmm.py` | Formatting-only observation-model reflow; `PitchIOIHMM.step` removes its local optimized implementation and calls the inherited forward step. | Formatting requires no port. Port the inherited optimized step into the shared Rust HMM helper so `pthmm` and `hmm` use the equivalent optimization. |
| `pyproject.toml` | Package version changes to 0.5.2 and minimum `python-hiddenmarkov` changes from 0.1.4 to 0.3.0. | Update and lock the Python reference environment; record the resolved dependency version in generated reference metadata. |

## R1: Pin the released reference and its updated dependency

- **Decision**: Pin pymatchmaker to the released `v0.5.2` tag in
  `tools/golden/pyproject.toml`; regenerate `tools/golden/uv.lock` so its
  `python-hiddenmarkov` version satisfies `>=0.3.0`. Update the release tags and version
  checks in both golden generators, and include the resolved hiddenmarkov version in
  reference metadata.
- **Rationale**: The repository's golden values are only meaningful when their package and
  dependency versions can be identified. The pymatchmaker 0.5.2 project metadata requires
  `python-hiddenmarkov>=0.3.0`; its release note attributes the `pthmm` speedup to that
  dependency.
- **Alternatives considered**: Keep python-hiddenmarkov 0.1.4 (rejected: violates the
  released pymatchmaker dependency floor and cannot exercise its optimized step). Pin an
  unreleased pymatchmaker commit (rejected: violates constitution principle I).
- **Risk**: Lock regeneration may move unrelated transitive packages. Review those changes
  against the lockfile diff and do not attribute resulting golden changes to pymatchmaker
  without evidence.

## R2: Port the optimized forward step into the shared banded HMM helper

The [python-hiddenmarkov 0.3.0 source](https://github.com/CarlosCancino-Chacon/hiddenmarkov/blob/v0.3.0/hiddenmarkov/__init__.py)
selects `forward_algorithm_step_optimized` by default for constant transition models. On
later observations it finds the first and last non-zero forward states, multiplies only the
corresponding transition rows, limits output columns to the last column reachable from those
rows, then multiplies by observation probabilities and normalizes by
`max(sum(forward_variable), 1e-6)`. The release documentation states the optimized and
standard operations are numerically equivalent up to floating-point summation order.

- **Decision**: Add the reachable-row / reachable-column path to the existing `Banded`
  forward-step implementation in `src/prob/hidden_markov.rs`. Keep the initial distribution
  on the first observation, preserve ascending transition-row accumulation, observation
  weighting, and the existing normalization guard. Both `pthmm` and `hmm` already call this
  shared function, so they receive the optimization without a new API or separate code path.
- **Rationale**: pymatchmaker 0.5.2 removes its local `PitchIOIHMM` fast path and delegates
  to the optimized dependency implementation. The same dependency default also accelerates
  `pthmm`, which is why the release specifically calls out that method. The Rust port
  already stores transitions as non-zero bands but currently visits every row and builds
  the full prior vector; limiting the work to reached states is the missing behavior.
- **Alternatives considered**: Keep the current full sparse-band traversal (rejected: it
  omits the release's principal performance improvement, especially for `pthmm`). Copy the
  Python matrix-slice/BLAS implementation literally (rejected: Rust already stores sparse
  transition bands, so a dense temporary matrix would add allocation and duplicate work).
- **Validation**: Compare optimized and full banded results for initial input, concentrated
  support, non-contiguous support, and all-zero forward variables. Then compare `pthmm` and
  `hmm` goldens within their existing tolerances and measure both on long-score fixtures.
- **Risk**: The Python slice dot product may accumulate floating-point values in a different
  order from the Rust sparse-row loop. Existing documented parity tolerances remain the
  limit; a tolerance failure requires investigating arithmetic order, not silently loosening
  the tests.

## R3: Release behavior is internal; keep user-facing method contracts stable

- **Decision**: Do not add or remove a method, configuration key, CLI option, or Python
  binding. The release changes the implementation and dependency used by existing HMM
  methods, not their public configuration or result shape.
- **Rationale**: The complete pymatchmaker release diff contains no `methods.yaml` change and
  no new public matcher capability. Both Rust HMMs already share `Banded` and expose the
  same positions, alignment paths, and debug probability vectors.
- **Alternatives considered**: Expose an optimization toggle (rejected: the reference
  chooses the optimized path by default, the user did not request configurability, and a
  second mode would complicate future syncs).

## R4: Audit all current pymatchmaker references without rewriting history

- **Decision**: Audit current release pins, generator checks/metadata, reference tables,
  active API and behavior descriptions, tests, and sync procedure. Update current references
  to 0.5.2 and record python-hiddenmarkov 0.3.0+ where it describes the active environment.
  Keep old feature specifications, old release-to-release comparisons, and asset provenance
  unchanged when they accurately describe past work.
- **Rationale**: The repository-wide search finds active 0.5.1 references in `README.md`,
  `UPSTREAM.md`, `tools/golden/pyproject.toml`, `tools/golden/uv.lock`, and both golden
  generators; the `0.5.1` mention in `specs/005-upstream-0.5-sync/` is historical. The
  reported `Cargo.lock` version 0.5.0 is a different package and must not be changed based on
  a substring match.
- **Alternatives considered**: Replace every textual occurrence of 0.5.1 (rejected: it
  would falsify historical specifications and change unrelated records).
- **Guidance update**: Revise `UPSTREAM.md`'s procedure so it starts from the complete
  upstream release diff, assesses every changed file, maps relevant behavior to code/tests or
  a reasoned exclusion, and only then uses the module map to locate existing port
  counterparts. The current instruction that "Only the files in the table matter" is
  inconsistent with the requested completeness.

## R5: Test and performance evidence

- **Decision**: Regenerate MIDI and audio reference outputs from the locked environment,
  inspect every changed golden, run the workspace test suite and existing quality gates, and
  compare benchmark medians over at least five runs. Record the benchmark delta for `pthmm`
  and `hmm` as well as overall existing methods.
- **Rationale**: Principle I requires reference parity; principle III requires benchmark
  evidence for algorithm changes and flags regressions above 10%.
- **Alternatives considered**: Treat successful compilation as sufficient (rejected: it does
  not verify output parity or the intended performance effect).

## Validation results

- The locked environment resolves pymatchmaker `0.5.2` and python-hiddenmarkov `0.3.0`.
- The focused forward-step tests and `cargo test --workspace --exclude matchmaker-py` pass;
  the Python binding crate also compiles with `cargo test -p matchmaker-py`.
- All MIDI golden methods were regenerated, including the opt-in Parangonar methods. The
  workspace golden tests pass. Only one HMM probability array changed numerically; the
  existing golden tests accept it without changing their tolerances.
- Five-run Rust measurements on `tools/bench/bundled.toml`, compared with the pre-change
  `HEAD` worktree, show median inference changes on the three fixtures:

  | Method | Simple Mozart | Mozart | Bach fugue | Median across fixtures |
  |---|---:|---:|---:|---:|
  | `arzt` | -3.77% | -19.35% | +0.00% | -3.77% |
  | `dannenberg` | +0.00% | -18.18% | -10.00% | -10.00% |
  | `dixon` | -3.09% | -21.90% | -6.57% | -6.57% |
  | `hmm` | -0.88% | -18.12% | -37.86% | -18.12% |
  | `outerhmm` | -8.33% | -16.67% | -27.47% | -16.67% |
  | `pthmm` | +3.06% | -16.38% | -27.18% | -16.38% |
  | All six methods | - | - | - | -14.89% median inference latency |

  Negative values are faster. No per-method median regression exceeded 10%. The reference
  benchmark's default method list fails on `pfkorz` with
  `KeyError: 'ParticleFilterMidiProcessor'`; the five-run report therefore uses the six
  other methods (including both HMMs) on the same bundled manifest.
- Audio goldens were regenerated with the pinned SoundFont and native FluidSynth 2.6.2
  after adding its `bin` directory to `PATH`. All 581 generated files matched the existing
  file set; all non-JSON files were byte-identical. The only changes were reference fields
  in 62 JSON files, each recording pymatchmaker 0.5.2, python-hiddenmarkov 0.3.0 and the
  current lockfile digest. The upstream 0.5.2 diff does not change audio paths.
- `cargo fmt --check`, Clippy, the wasm build, `cargo deny check`, and the Rust 1.88 MSRV
  check pass. `cargo deny check` emits existing duplicate-`syn` and unused license-allowance
  warnings.
