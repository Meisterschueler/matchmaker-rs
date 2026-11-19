# Audio fixture provenance

## SoundFont prerequisite (T001: verified)

The runtime and score-reference generator require an explicitly configured
local SF2 SoundFont. No SoundFont is bundled or downloaded automatically. Do not
generate or distribute SoundFont-derived fixtures until the archive contents,
license, and SHA-256 have been verified and recorded here.

### Downloaded candidate

- Asset: FreePats **Upright Piano KW**, archive/build version **2019-07-03**,
  small SF2 variant. The current publisher page says 2022-02-21; the downloaded
  archive identifies itself as 2019-07-03, so these versions must not be conflated.
- Publisher: https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html
- Archive: `UprightPianoKW-small-SF2-20190703.7z`, 6,093,984 bytes,
  SHA-256 `3BD025E7C2FFA9E6F3F99215CE383C9CEBBAC991CFBF9BE1453CDB4328EC3492`.
- Extracted SF2: `UprightPianoKW-small-20190703.sf2`, 9,456,310 bytes,
  SHA-256 `CF2A98EB38A32C4954B4B6E2CAAE4112D62DD8E892ECEEFDD7942B0E7D01AC2F`.
- The archive contains `readme.txt` and `cc0.txt`. Its readme identifies the
  sound bank as CC0 and documents a Kawai upright piano recorded in January 2017
  by Gonzalo (`humanogonzalo@gmail.com`) and Roberto (`roberto@zenvoid.org`) using
  a Zoom H1; Roberto cropped, edited, and processed the recordings.
- Archive integrity check: 7-Zip test passed. Preserve both included text files
  beside the SF2.
- Attribution: FreePats Upright Piano KW; recordings by Gonzalo and Roberto.
  CC0 does not require attribution, but retain this provenance with fixtures.
- Source pin: FreePats' Upright Piano KW page and the archive's included readme;
  the archive filename and SHA-256 above identify this exact 2019-07-03 asset.
  The page currently labels the bank 2022-02-21, so provision only the archive
  whose extracted readme identifies 2019-07-03 and whose hash matches this pin.

### Provisioning rules

Keep the archive and extracted files outside tracked fixture directories, for
example in `target/audio-assets/`. Pass the local SF2 path explicitly to the
audio constructor, `--soundfont-path`, or Python `soundfont_path` option. The
generator verifies the pinned SF2 hash and records it in score-derived metadata.
The CI workflows download the exact pinned `.7z` archive from its fixed HTTPS
URL, verify the archive hash, extract it under `target/audio-assets/`, and
verify the extracted SF2 hash. The general CI workflow distributes the
verified SF2 to platform and Python test jobs as a short-lived artifact; it is
not committed to the repository or included in benchmark reports.

### Verification status (2026-11-01)

The user supplied the archive in `target/audio-assets/`. The archive contents,
included CC0 text, and both hashes above were inspected locally; its 7-Zip
integrity test passed. The archive's version is explicitly pinned as 2019-07-03,
not the 2022-02-21 date currently shown on the publisher page. Files remain
outside tracked fixture directories. T001 is complete; the T003 generator uses
this exact pin and records its SF2 hash in generated fixture metadata.

### Reference generation

`tools/golden/generate_audio.py` uses pymatchmaker v0.4.0 and the locked
`tools/golden/uv.lock`. Its `--processors-only` mode generates a deterministic
three-tone input and processor goldens for all four method/processor pairs under
`tests/golden/audio/processors/`, without requiring FluidSynth. The checked-in
arrays include the input samples, feature rows, times, and reference metadata.

Its `--codec-fixtures-only` mode writes small synthetic mono WAV, FLAC, MP3, and
Ogg/Vorbis files under `tests/data/audio/codec_samples/` for offline decoder
tests. They contain generated 440/880-Hz tones and have no third-party recording
provenance. Their metadata and SHA-256 values are recorded in
`tests/data/audio/codec_samples/codec_fixtures.json` and covered by
`tests/audio_io.rs`:

| File | Bytes | SHA-256 |
|---|---:|---|
| `decoder_tone.wav` | 44,144 | `245819C5D8614D69CCB38D442629122F0E11BD5E35470F394CAD9935D2E6373D` |
| `decoder_tone.flac` | 8,607 | `0C7D25DD84209DF59FD959971BAA3A593D1B393C43C264015BD8953694F1C74E` |
| `decoder_tone.mp3` | 3,437 | `82344256A50D8DDFEBE32C3BFB02CC50206E35E36522D00489FAEC600B26CABD` |
| `decoder_tone.ogg` | 4,818 | `4CD531206629D4175DABDFE0AF981E98965FA1071AB8E7B032C9CE06AD5AECBA` |

The Ogg encoder may vary container bytes between runs. The generator preserves
existing files and records their hashes; replace a fixture intentionally and
update this manifest/table together.

The full mode writes one directory per method/processor pair under
`tests/golden/audio/`, containing performance features and times, score features
and frame-to-beat mapping, positions, alignment path, and metadata with the
reference, FluidSynth version, and SoundFont hashes. The SoundFont path and
SHA-256 are checked before use. The reference score renderer is patched only to
pass this pinned SF2 to Partitura; its timing, padding, and cropping logic
remains the reference implementation.

The optional `--render-performance` flag writes a float WAV derived from the
score and pinned SF2 to `tests/data/audio/simple_mozart_k265_var1_sf2.wav`.
Reference rendering requires Partitura's optional FluidSynth support and a
working native FluidSynth library; neither is needed by the Rust runtime. The
checked-in T011 fixture was rendered with FluidSynth 2.6.2 from the official
Windows x64 release archive, SHA-256
`808295345d43c9c428f3fa87d29fbf97a9926f356df95559ff2689f3bd4072d7`. The WAV
is 1,340,698 bytes with SHA-256
`32DD5C2ED10040658DFAA09ABFE314F97DCE35638C62027216CACA7B31EAAA60`. Its four
reference runs are checked in under `tests/golden/audio/`; metadata records
FluidSynth, pymatchmaker, Partitura, and SoundFont versions/hashes.

### Tempo-aware Arzt references

`generate_audio.py --tempo-replay-only` writes fixed, synthetic feature-vector cases
under `tests/golden/audio/tempo_replay/` (steady, accelerating, slowing). The cases
record eight tempo-state columns, every MT19937 uniform draw, final altered score
features/mapping, and exact reset reproducibility with seed 1984. They contain no
third-party recording or SoundFont-derived samples.

`generate_audio.py --arzt-tempo-only` writes separate Chroma/LSE references under
`tests/golden/audio/tempo_pipeline/` using the existing verified WAV and SF2.
Initial score frames and mappings are saved before the tempo model alters them.
The same FluidSynth 2.6.2 archive pin above applies. On Windows, put its `bin`
directory on PATH and in the DLL search path before importing Partitura.

The Rust score renderer explicitly initializes CC91 to zero. The reference
FluidSynth channel has CC91=0, whereas RustySynth initializes it to 40; accepting
that default introduces additional reverb and changes the score reference.
This is renderer calibration, not a relaxed test tolerance or a new native runtime dependency.

The Rust score renderer also peak-normalizes the rendered score to 1.0, because
Partitura's FluidSynth export does (`synthesize_fluidsynth` divides the piece by
its peak). RustySynth returns its raw level (peak about 0.14), which is 17 dB below
the reference in every band; chroma and LSE hide that through per-frame
normalization, unnormalized features such as `raw_spectrum` do not. Score features
are extracted in one whole-signal pass like the reference. Both conventions are
documented, with their known limits, in research R11 of `specs/004-audio-methods`.

### Repeat / volta audio fixtures

`tests/data/fixture_repeats_voltas.musicxml` is a small hand-authored synthetic
score. `tools/golden/generate_audio.py --multi-fold-only` renders its taken- and
skipped-repeat variants with the pinned CC0 SoundFont above and records Partitura
0.4.0 candidate/replay goldens under `tests/golden/audio/multi_fold/`.
The score SHA-256 is `44E765CBD6CA3C48871123F290CC6B360234855C8599207DFC0D358B8809D4E9`.

| File | Bytes | SHA-256 |
|---|---:|---|
| `arzt_multi_fold/repeat_taken.wav` | 2,134,498 | `F35C15390B45259DBAA9A72421FD0141871705749911496549DF497A18061F07` |
| `arzt_multi_fold/repeat_skipped.wav` | 1,193,694 | `6F7D56C9D82778DCB10C8EE412C99F3EF6EE79EBA89AF9CCFE91F560CD97668E` |

These are score-derived synthetic renders, not third-party recordings. Their
metadata pins the source score hash, SoundFont hash, reference lock, and FluidSynth
version. The matching `.tsv` files contain unique note-onset times and folded-score
beats for metric evaluation, not one row for every intermediate audio frame.

`ground_truth/score_onsets.tsv` is an eight-onset direct TSV fixture generated from
the bundled score's locked-reference frame-to-beat mapping. Its score, performance,
and CC0 SoundFont hashes and reference metric golden are under
`tests/golden/audio/ground_truth/`.

### SKF fixed-template replays

`tools/golden/generate_audio.py --skf-only` (needs FluidSynth and the pinned SoundFont above)
writes `tests/golden/audio/skf/` (about 420 KB, deterministic): `synthetic` (eight chords with
Gaussian templates, a beam of 12 hypotheses, one silent and one half-negative frame), `override`
(16 kHz, FFT 1024, hop 512), `one_chord`, and `bundled` (the bundled score with the reference's
synthesized templates and 90 raw-spectrum frames of the bundled performance). Each case stores
the inputs both sides consume (score table, templates, observations), the reference's chord
sequence, positions, chord path and the hypotheses `(chord, age, probability, tempo mean and
variance)` after selected frames, so `tests/golden_audio_skf.rs` isolates the recurrence from
template synthesis. The `bundled` metadata records the SoundFont, performance and FluidSynth
identities; its templates are the reference's float32-rounded spectra of chords rendered from the
CC0 SF2 (derived spectra, no samples). The generator patches `partitura.save_wav_fluidsynth` with
the pinned SF2 and fails if the reference falls back to Gaussian peaks: unpatched, the reference
would do so silently here because Partitura's bundled MuseScore file is an empty placeholder
(research R12). `tools/golden/compare.py tests/golden/audio/skf/bundled <export>` checks the
template mode, chord model and templates of a `follow --method skf --export` run.

### CQT references

`tools/golden/generate_audio.py --cqt-only` (locked reference environment, no FluidSynth) writes
`tests/golden/audio/processors/cqt/` (98 files, about 800 KB, deterministic): `default`
(44.1 kHz, hop 1470), `override` (16 kHz, hop 640), `tones` (odd hop, no decimation), `early`
(hop 2048, early downsampling), `norm2`, `silent`, `short`, and `performance` (the rows of the
bundled performance WAV above, which is the input; no copy of the samples). The other cases store
the stream rows, the whole-signal rows, the `soxr` decimation outputs and the sparse kernels of the
reference octaves. The comparison limits, including the frame-maximum term that `float32` and
`soxr` noise make necessary, are in research R13 of `specs/004-audio-methods` and in every
`meta.json`. `tools/golden/compare.py tests/golden/audio/processors/cqt/performance <export>`
checks the export of `matchmaker features tests/data/audio/simple_mozart_k265_var1_sf2.wav
--processor cqt --export <dir>` against the reference rows.

### CQT spectral-flux references

`tools/golden/generate_audio.py --cqt-spectral-flux-only` (locked reference environment, no
FluidSynth) writes `tests/golden/audio/processors/cqt_spectral_flux/` (36 files, about 420 KB,
deterministic): `default` (piano, 44.1 kHz, hop 1470), `override` (16 kHz, hop 640, the
Audio-OuterHMM rates), `melody` (synthetic repeated and changing pitches with silences, hop 882),
`silent`, `short` and `performance` (the rows of the bundled WAV, no copy of the samples). Each
case stores the stream rows (89 values, and 88 without flux for the first three), the dropped first
call (flux 0) and a whole-signal call (a fresh processor, flux 0); `default` also records the
lifecycle: `reset()` keeps the history, and a stream call after the whole-signal call fails with
NumPy's broadcast error. The limit is the cqt one, with the frame maximum over the 88 bins.
`compare.py tests/golden/audio/processors/cqt_spectral_flux/performance <export>` checks the export
of `matchmaker features ... --processor cqt_spectral_flux --export <dir>`.

### Audio outer-product HMM templates

`tools/golden/generate_audio.py --outerhmm-only` writes fixed-template reference replays under
`tests/golden/audio/outerhmm/` and embeds the exact learned archive from the locked pymatchmaker
0.4.0 package at `src/prob/assets/pitch_templates_16000_640.npz`.

- Upstream: `matchmaker/prob/templates/pitch_templates_16000_640.npz`, introduced by commit
  `b7955e645e3998e4d4b5deed7370948991766e4e`; Git blob
  `36b3ca1a8f86760e3d4b5deed7370948991766e4e`.
- SHA-256: `7448383AA9C7B4B49B3A4A52B8BED2AFF994B0F3266BA0372A28738A3DDE8A03`.
- Runtime SoundFont provenance: the pinned CC0 SF2 SHA-256 is
  `CF2A98EB38A32C4954B4B6E2CAAE4112D62DD8E892ECEEFDD7942B0E7D01AC2F`; each run export
  records the actual configured SoundFont hash in `score/meta.json`.
- License: Apache-2.0 under the upstream repository license; there is no separate license or
  training-corpus statement for the archive. The corpus is therefore recorded as unidentified.
- Arrays: `mu` and `var`, both 88x88 little-endian float64; `mu` is Fortran ordered. The released
  overlay replaces 77 rows whose sum exceeds 0.5. `var` is not used by the follower.
- Remaining rows are synthesized from the pinned CC0 SF2 at three velocities, with a fresh
  RustySynth instance per velocity. Their measured minimum cosine similarity is 0.97678 and
  maximum total variation is 0.0923 against FluidSynth;
  fixed-template model replay and the full-pipeline evaluator/position/p95 gates are separate.
- The reference facade forwards hop_length but not sample_rate to the follower; a rate override
  changes processor rate and derived hop while follower templates retain 16000 Hz. The default
  16000/640 pair uses the learned overlay; other pairs synthesize all rows.

The golden cases cover weighted/sustained chords, nonzero repeat/skip probability, one chord, and
the bundled score/performance. The bundled replay uses the score note array and CQT-flux stream;
it does not render score audio or copy the performance samples.

### Rejected candidate

GeneralUser GS 2.0.3 was considered at upstream commit
`684543d5e5efaef08d02be50dcda8d552478fa60`. Its
[license](https://github.com/mrbumpy409/GeneralUser-GS/blob/684543d5e5efaef08d02be50dcda8d552478fa60/documentation/LICENSE.txt)
permits music production and use in software projects, but explicitly states
that the author cannot establish the origin of all contained samples. It was
not selected for redistributable fixtures under this feature's provenance gate.
