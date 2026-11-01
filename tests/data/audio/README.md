# Audio fixture provenance

## SoundFont prerequisite (T001: verified)

The runtime and fixture generator require an explicitly configured local SF2
SoundFont. No SoundFont is bundled or downloaded automatically. Do not generate
or distribute audio fixtures until the archive contents, license, and SHA-256
have been verified and recorded here.

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

### Outstanding verification and provisioning

1. Keep the archive and extracted files outside tracked fixture directories,
  for example in
   `target/audio-assets/`. Pass the local SF2 path explicitly to the future
   audio constructor, `--soundfont-path`, or Python `soundfont_path` option.
2. Pin the verified asset details in the fixture generator before creating
  reference outputs.
   Generated fixtures must record their score, generator, SoundFont hash, sample
   rate, and rendering settings. No fixtures or goldens exist here yet.

### Verification status (2026-11-01)

The user supplied the archive in `target/audio-assets/`. The archive contents,
included CC0 text, and both hashes above were inspected locally; its 7-Zip
integrity test passed. The archive's version is explicitly pinned as 2019-07-03,
not the 2022-02-21 date currently shown on the publisher page. Files remain
outside tracked fixture directories. T001 is complete; T003 may use this exact
pin and must record its SF2 hash in generated fixture metadata.

### Rejected candidate

GeneralUser GS 2.0.3 was considered at upstream commit
`684543d5e5efaef08d02be50dcda8d552478fa60`. Its
[license](https://github.com/mrbumpy409/GeneralUser-GS/blob/684543d5e5efaef08d02be50dcda8d552478fa60/documentation/LICENSE.txt)
permits music production and use in software projects, but explicitly states
that the author cannot establish the origin of all contained samples. It was
not selected for redistributable fixtures under this feature's provenance gate.
