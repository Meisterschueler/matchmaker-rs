# Audio API Contract

This contract adds prerecorded audio support while preserving the existing MIDI interfaces and defaults.

## Supported Inputs

- Score: existing MusicXML/XML/MXL or MIDI score formats.
- Performance audio: WAV, FLAC, MP3, or Ogg/Vorbis decoded by the selected pure-Rust backend.
- Input audio is converted to mono and resampled to the configured target `sample_rate` (44,100 Hz by default). Unsupported codecs/containers, including M4A/MP4 and AAC in the initial decoder set, return an actionable error; no system codec fallback is used.
- Live devices and streamed device capture are not part of this feature.
- Score rendering requires a readable SF2 SoundFont path. It must be pinned and license-checked before distribution. The CLI and Python facade accept `soundfont_path`; the Rust audio constructor accepts it in audio config.

## Method and Processor Defaults

| Input/method | Default processor | Sample rate | Frame rate / hop | Follower defaults |
|---|---|---:|---|---|
| Audio `arzt` | `chroma` | 44,100 Hz | 30 fps / 1,470 samples | `window_size=10` seconds, `start_window_size=0.1` seconds, `step_size=3`, Manhattan distance |
| Audio `dixon` | `lse` | 44,100 Hz | 50 fps / 882 samples | `window_size=10` seconds, `max_run_count=3`, Euclidean distance |

Supported audio processors are `chroma` and `lse`. Users may override the processor, target sample rate, frame rate or hop length, and method-specific parameters. `hop_length` overrides `frame_rate`; audio is resampled to the configured target before framing. The processor-method matrix and invalid combinations are validated at construction.

## Rust Library

Keep the current MIDI constructor and `Method` configuration behavior unchanged. Add an explicit audio entry point accepting a score path, performance-audio path, audio method/config, tempo/unfold settings, and SoundFont path. The returned run exposes per-frame positions and the existing `(performance seconds, score beats)` alignment path.

## CLI

Audio runs use the existing `follow` command with explicit input type and SoundFont:

```text
matchmaker follow SCORE PERFORMANCE --input-type audio --method arzt --soundfont-path soundfont.sf2 --output alignment.tsv --export export/
```

`--processor` selects `chroma` or `lse`; repeatable `--set KEY=VALUE` accepts documented audio and follower parameters. The existing two-positional-path MIDI invocation and its defaults remain valid.

## Python

Use the existing constructor shape with `input_type="audio"` and the performance-file path:

```python
Matchmaker(
    "score.musicxml",
    "performance.wav",
    input_type="audio",
    method="arzt",
    kwargs={"soundfont_path": "soundfont.sf2"},
)
```

The audio `arzt` and `dixon` methods yield score positions and expose their final alignment path using the same shape as the existing Python facade. `processor=` overrides the method default. Invalid method, processor, or parameter values raise `ValueError`; file/decode/render failures identify the failed path.

## Exports and Parity

Exports contain decoded/frame timing metadata, processor features, score reference features, the reference-frame-to-beat mapping, per-input-frame positions, and the final alignment path. Arrays are NumPy-readable. Stage comparison tests cover both processors and all four method/processor combinations; end-to-end renderer deviations are reported with the first differing frame and evaluation metrics.
