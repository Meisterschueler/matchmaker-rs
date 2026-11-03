"""Generate audio fixtures and reference outputs for matchmaker-rs.

Run from ``tools/golden`` with the locked environment. The reference renderer
uses Partitura/FluidSynth with the pinned local SF2; the Rust runtime remains
pure Rust and uses RustySynth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from ctypes import c_int
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import matchmaker
import numpy as np
import partitura
import soundfile as sf
from matchmaker import Matchmaker
from matchmaker.features.audio import (
    SAMPLE_RATE,
    ChromagramProcessor,
    LogSpectralEnergyProcessor,
)
from matchmaker.dp.oltw_arzt import OnlineTimeWarpingArztFrame
from matchmaker.matchmaker import DEFAULT_TEMPO
from matchmaker.utils.tempo_models import ArztTempoModel
from matchmaker.utils import misc as reference_misc
from scipy.io import wavfile

from generate import load_score_like_matchmaker

REFERENCE_TAG = "v0.4.0"
SOUNDFONT_SHA256 = "CF2A98EB38A32C4954B4B6E2CAAE4112D62DD8E892ECEEFDD7942B0E7D01AC2F"
METHODS = {
    "arzt": {
        "frame_rate": 30,
        "window_size": 10,
        "start_window_size": 0.1,
        "step_size": 3,
    },
    "dixon": {"frame_rate": 50, "window_size": 10, "max_run_count": 3},
}
PROCESSORS = ("chroma", "lse")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCORE = ROOT / "tests/data/simple_mozart_k265_var1.musicxml"
DEFAULT_PERFORMANCE = ROOT / "tests/data/audio/simple_mozart_k265_var1_sf2.wav"
DEFAULT_SOUNDFONT = (
    ROOT
    / "target/audio-assets/UprightPianoKW-small-SF2-20190703/"
    / "UprightPianoKW-small-20190703.sf2"
)
DEFAULT_OUTPUT = ROOT / "tests/golden/audio"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_array(path: Path, values, dtype) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(values, dtype=dtype))


def reference_info() -> dict:
    lockfile = Path(__file__).with_name("uv.lock")
    return {
        "package": "pymatchmaker",
        "tag": REFERENCE_TAG,
        "version": matchmaker.__version__,
        "partitura": version("partitura"),
        "numpy": np.__version__,
        "uv_lock_sha256": sha256(lockfile),
    }


def fluidsynth_version() -> str:
    import fluidsynth

    major, minor, micro = c_int(), c_int(), c_int()
    fluidsynth.fluid_version(major, minor, micro)
    return f"{major.value}.{minor.value}.{micro.value}"


@contextmanager
def use_soundfont(soundfont: Path):
    original_renderer = partitura.save_wav_fluidsynth

    def pinned_renderer(*args, **kwargs):
        kwargs["soundfont"] = str(soundfont)
        return original_renderer(*args, **kwargs)

    with patch.object(partitura, "save_wav_fluidsynth", pinned_renderer):
        yield


def render_with_soundfont(score_part, tempo: float, soundfont: Path) -> np.ndarray:
    """Use the reference timing/padding rules with the explicitly pinned SF2."""
    try:
        with use_soundfont(soundfont):
            audio = reference_misc.generate_score_audio(
                score_part, tempo, SAMPLE_RATE
            )
    except ImportError as error:
        raise RuntimeError(
            "Reference audio generation requires Partitura's optional "
            "FluidSynth support and a working FluidSynth library."
        ) from error
    return np.asarray(audio, dtype=np.float32)


def render_performance(score_path: Path, performance_path: Path, soundfont: Path) -> None:
    _, score_part, _ = load_score_like_matchmaker(score_path)
    from matchmaker.utils.misc import get_tempo_from_score

    score_tempo = get_tempo_from_score(score_part, str(score_path))
    tempo = float(score_tempo) if score_tempo is not None else float(DEFAULT_TEMPO)
    audio = render_with_soundfont(score_part, tempo, soundfont)
    performance_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(performance_path), SAMPLE_RATE, audio)


def generate_processor_goldens(output: Path) -> None:
    """Generate processor-only references without requiring score rendering."""
    sample_rate = SAMPLE_RATE
    duration = 0.4
    times = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
    samples = (
        0.35 * np.sin(2 * np.pi * 440.0 * times)
        + 0.2 * np.sin(2 * np.pi * 659.2551138257398 * times)
        + 0.1 * np.sin(2 * np.pi * 783.9908719634985 * times)
    ).astype(np.float32)
    base = output / "processors"
    write_array(base / "input_samples.npy", samples, "<f4")

    for method, method_defaults in METHODS.items():
        frame_rate = method_defaults["frame_rate"]
        hop_length = sample_rate // frame_rate
        for processor_name in PROCESSORS:
            if processor_name == "chroma":
                processor = ChromagramProcessor(
                    sample_rate=sample_rate,
                    hop_length=hop_length,
                )
            else:
                processor = LogSpectralEnergyProcessor(
                    sample_rate=sample_rate,
                    hop_length=hop_length,
                    norm=2,
                )
            cache_size = processor.n_fft - hop_length
            padded = np.pad(
                samples,
                (0, (-len(samples)) % hop_length),
                mode="constant",
            )
            cache = np.zeros(cache_size, dtype=np.float32)
            emitted = 0
            feature_rows, feature_times = [], []
            for chunk_index, hop in enumerate(
                padded.reshape(-1, hop_length)
            ):
                block = np.concatenate((cache, hop))
                features, frame_time = processor(
                    (block, emitted * hop_length / float(sample_rate))
                )
                if chunk_index > 0:
                    feature_rows.extend(np.asarray(features, dtype=np.float32))
                    feature_times.extend([float(frame_time)] * len(features))
                    emitted += 1
                if cache_size:
                    cache = block[-cache_size:].copy()

            directory = base / f"{method}_{processor_name}"
            write_array(
                directory / "features.npy",
                np.asarray(feature_rows, dtype=np.float32).reshape(
                    -1, 12 if processor_name == "chroma" else processor.dim
                ),
                "<f4",
            )
            write_array(directory / "times.npy", feature_times, "<f8")
            write_json(
                directory / "meta.json",
                {
                    "reference": reference_info(),
                    "generator": "tools/golden/generate_audio.py",
                    "input": "generated three-tone signal",
                    "sample_rate": sample_rate,
                    "frame_rate": frame_rate,
                    "hop_length": hop_length,
                    "method": method,
                    "processor": processor_name,
                    "frames": len(feature_rows),
                    "tolerances": {"features_rtol": 2e-4, "features_atol": 1e-6},
                },
            )
            print(f"{method}/{processor_name}: {len(feature_rows)} frames")


def generate_tempo_replay_goldens(output: Path) -> None:
    """Record seeded tempo tracking independently of audio synthesis."""
    reference_frames = 480
    frame_rate = 30
    indices = np.arange(reference_frames)
    reference = np.zeros((reference_frames, 12), dtype=np.float32)
    reference[indices, (indices // 12) % 12] = 1.0 + (indices % 12) / 20.0
    frame_to_beat = indices.astype(np.float64) / 12.0
    score_positions = np.arange(40, dtype=np.float32)

    class RecordingRandom:
        def __init__(self, rng):
            self.rng = rng
            self.draws = []

        def uniform(self, low, high):
            value = self.rng.uniform(low, high)
            self.draws.append(float(value))
            return value

    for name, lengths in (
        ("steady", (160, 160, 160)),
        ("accelerating", (220, 160, 110)),
        ("slowing", (110, 160, 220)),
    ):
        replay_indices = np.concatenate([
            np.linspace(start, start + 159, length).astype(np.int64)
            for start, length in zip((0, 160, 320), lengths)
        ])
        observations = reference[replay_indices].copy()
        times = np.arange(len(observations), dtype=np.float64) / frame_rate
        follower = OnlineTimeWarpingArztFrame(
            reference_features=reference.copy(),
            score_positions=score_positions.copy(),
            ref_frame_to_beat=frame_to_beat.copy(),
            frame_rate=frame_rate,
            window_size=10,
            start_window_size=0.1,
            step_size=3,
            tempo_model=ArztTempoModel,
        )

        def replay():
            random = RecordingRandom(follower.tempo_model._rng)
            follower.tempo_model._rng = random
            positions, states, draws = [], [], []
            for observation, perf_time in zip(observations, times):
                before = len(random.draws)
                positions.append(follower(observation, float(perf_time)))
                model = follower.tempo_model
                states.append([
                    follower._current_frame,
                    follower.current_index,
                    model.current_tempo,
                    model.beat_period,
                    model.n_inserted,
                    model.n_deleted,
                    model.n_postponed,
                    follower.N_ref,
                ])
                draws.append(random.draws[before:])
            return np.asarray(positions), np.asarray(states), draws

        positions, states, draws = replay()
        final_features = follower.reference_features.copy()
        final_mapping = follower._ref_frame_to_beat.copy()
        follower.reset()
        repeated_positions, repeated_states, repeated_draws = replay()
        np.testing.assert_array_equal(positions, repeated_positions)
        np.testing.assert_array_equal(states, repeated_states)
        assert draws == repeated_draws
        assert np.isfinite(states).all()

        directory = output / "tempo_replay" / name
        for filename, values, dtype in (
            ("score_features", reference, "<f4"),
            ("score_positions", score_positions, "<f4"),
            ("score_frame_to_beat", frame_to_beat, "<f8"),
            ("performance_features", observations, "<f4"),
            ("performance_times", times, "<f8"),
            ("positions", positions, "<f8"),
            ("tempo_states", states, "<f8"),
            ("final_score_features", final_features, "<f4"),
            ("final_score_frame_to_beat", final_mapping, "<f8"),
        ):
            write_array(directory / f"{filename}.npy", values, dtype)
        write_json(directory / "meta.json", {
            "reference": reference_info(),
            "generator": "tools/golden/generate_audio.py",
            "input": "synthetic pitched feature vectors; no rendered audio",
            "method": "arzt_tempo",
            "frame_rate": frame_rate,
            "window_size": 10,
            "start_window_size": 0.1,
            "step_size": 3,
            "random_seed": 1984,
            "frames": len(observations),
            "state_columns": [
                "current_frame", "current_index", "relative_tempo", "beat_period",
                "n_inserted", "n_deleted", "n_postponed", "reference_frames",
            ],
            "uniform_draws_per_step": draws,
            "reset_repeat": "exact",
        })
        print(f"arzt_tempo/{name}: {len(positions)} frames, "
              f"inserted={int(states[-1, 4])}, deleted={int(states[-1, 5])}")


def generate_codec_fixtures(output: Path) -> None:
    """Write tiny, synthetic mono assets for the supported decoder containers."""
    sample_rate = 44_100
    times = np.arange(sample_rate // 2, dtype=np.float64) / sample_rate
    samples = (
        0.25 * np.sin(2 * np.pi * 440.0 * times)
        + 0.1 * np.sin(2 * np.pi * 880.0 * times)
    ).astype(np.float32)
    output.mkdir(parents=True, exist_ok=True)
    assets = []
    for extension, file_format, subtype in (
        ("wav", "WAV", "PCM_16"),
        ("flac", "FLAC", "PCM_16"),
        ("mp3", "MP3", None),
        ("ogg", "OGG", "VORBIS"),
    ):
        path = output / f"decoder_tone.{extension}"
        kwargs = {"format": file_format}
        if subtype is not None:
            kwargs["subtype"] = subtype
        if not path.exists():
            sf.write(path, samples, sample_rate, **kwargs)
        asset = {
            "file": path.name,
            "format": file_format,
            "sample_rate": sample_rate,
            "channels": 1,
            "samples": len(samples),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        assets.append(asset)
        print(f"{path.name}: {asset['bytes']} bytes, {asset['sha256']}")
    write_json(
        output / "codec_fixtures.json",
        {
            "generator": "tools/golden/generate_audio.py",
            "reference": reference_info(),
            "signal": "synthetic 440-Hz and 880-Hz sine tones",
            "assets": assets,
        },
    )


def record_follower(follower, recorder: dict) -> None:
    base = type(follower)

    def recorded_call(self, observation, perf_time, *args, **kwargs):
        position = base.__call__(self, observation, perf_time, *args, **kwargs)
        recorder["features"].append(
            np.asarray(observation, dtype=np.float32).reshape(-1).copy()
        )
        recorder["times"].append(float(perf_time))
        recorder["positions"].append(float(position))
        if "tempo_states" in recorder:
            model = self.tempo_model
            recorder["tempo_states"].append([
                self._current_frame, self.current_index, model.current_tempo,
                model.beat_period, model.n_inserted, model.n_deleted,
                model.n_postponed, self.N_ref,
            ])
        return position

    follower.__class__ = type(
        f"Recording{base.__name__}", (base,), {"__call__": recorded_call}
    )


def generate_case(
    score_path: Path,
    performance_path: Path,
    output: Path,
    soundfont_hash: str,
    soundfont: Path,
    method: str,
    processor: str,
) -> None:
    recorder = {"features": [], "times": [], "positions": []}
    if method == "arzt_tempo":
        recorder["tempo_states"] = []
    defaults = METHODS["arzt"] if method == "arzt_tempo" else METHODS[method]
    kwargs = {"sample_rate": SAMPLE_RATE, "processor": processor}
    kwargs.update(defaults)
    if method == "arzt_tempo":
        kwargs["tempo_model"] = ArztTempoModel

    with use_soundfont(soundfont):
        mm = Matchmaker(
            score_file=score_path,
            performance_file=performance_path,
            input_type="audio",
            method=method,
            processor=processor,
            kwargs=kwargs,
        )
        record_follower(mm.score_follower, recorder)
        initial_reference = mm.reference_features.copy()
        initial_mapping = mm.ref_frame_to_beat().copy()
        yielded = list(mm.run(verbose=False))

    if not np.array_equal(np.asarray(yielded), np.asarray(recorder["positions"])):
        raise RuntimeError("Recorded frame positions differ from yielded positions")

    features = (
        np.stack(recorder["features"])
        if recorder["features"]
        else np.zeros((0, 0), dtype=np.float32)
    )
    case = output / f"{method}_{processor}"
    write_array(case / "performance_features.npy", features, "<f4")
    write_array(case / "performance_times.npy", recorder["times"], "<f8")
    write_array(case / "score_features.npy", initial_reference, "<f4")
    write_array(case / "score_frame_to_beat.npy", initial_mapping, "<f8")
    write_array(case / "score_positions.npy", mm.score_positions, "<f8")
    write_array(case / "positions.npy", recorder["positions"], "<f8")
    write_array(case / "alignment_path.npy", mm.score_follower.alignment_path, "<f8")
    if method == "arzt_tempo":
        write_array(case / "tempo_states.npy", recorder["tempo_states"], "<f8")
        write_array(case / "final_score_features.npy", mm.score_follower.reference_features, "<f4")
        write_array(case / "final_score_frame_to_beat.npy", mm.score_follower._ref_frame_to_beat, "<f8")
    write_json(
        case / "meta.json",
        {
            "reference": reference_info(),
            "fluidsynth": fluidsynth_version(),
            "generator": "tools/golden/generate_audio.py",
            "score": score_path.name,
            "performance": performance_path.name,
            "soundfont_sha256": soundfont_hash,
            "sample_rate": mm.sample_rate,
            "frame_rate": mm.frame_rate,
            "hop_length": mm.hop_length,
            "method": method,
            "processor": processor,
            "follower_config": defaults,
        },
    )
    print(f"{method}/{processor}: {len(recorder['positions'])} frames")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", type=Path, default=DEFAULT_SCORE)
    parser.add_argument("--performance", type=Path, default=DEFAULT_PERFORMANCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--soundfont", type=Path, default=DEFAULT_SOUNDFONT)
    parser.add_argument(
        "--render-performance",
        action="store_true",
        help="render the default performance WAV from the score using the pinned SF2",
    )
    parser.add_argument(
        "--processors-only",
        action="store_true",
        help="generate synthetic chroma/LSE processor goldens without FluidSynth",
    )
    parser.add_argument(
        "--codec-fixtures-only",
        action="store_true",
        help="write synthetic WAV/FLAC/MP3/Ogg decoder test assets",
    )
    parser.add_argument(
        "--tempo-replay-only",
        action="store_true",
        help="generate seeded arzt_tempo replay goldens without score rendering",
    )
    parser.add_argument(
        "--arzt-tempo-only",
        action="store_true",
        help="generate only arzt_tempo audio pipeline goldens under tempo_pipeline/",
    )
    args = parser.parse_args()
    if args.tempo_replay_only:
        generate_tempo_replay_goldens(args.out.resolve())
        return
    if args.codec_fixtures_only:
        generate_codec_fixtures(ROOT / "tests/data/audio/codec_samples")
        return
    if args.processors_only:
        generate_processor_goldens(args.out.resolve())
        return

    score_path = args.score.resolve()
    performance_path = args.performance.resolve()
    soundfont = args.soundfont.resolve()
    output = args.out.resolve()

    if not score_path.is_file():
        parser.error(f"score does not exist: {score_path}")
    if not soundfont.is_file():
        parser.error(f"SoundFont does not exist: {soundfont}")
    soundfont_hash = sha256(soundfont)
    if soundfont_hash != SOUNDFONT_SHA256:
        parser.error(
            f"unexpected SoundFont SHA-256 {soundfont_hash}; "
            f"expected pinned asset {SOUNDFONT_SHA256}"
        )
    if args.render_performance:
        render_performance(score_path, performance_path, soundfont)
    if not performance_path.is_file():
        parser.error(
            f"performance WAV does not exist: {performance_path}; "
            "pass --render-performance to create it from the score"
        )

    if args.arzt_tempo_only:
        for processor in PROCESSORS:
            generate_case(
                score_path, performance_path, output / "tempo_pipeline",
                soundfont_hash, soundfont, "arzt_tempo", processor,
            )
        return

    for method in METHODS:
        for processor in PROCESSORS:
            generate_case(
                score_path,
                performance_path,
                output,
                soundfont_hash,
                soundfont,
                method,
                processor,
            )

    write_json(
        output / "manifest.json",
        {
            "reference": reference_info(),
            "fluidsynth": fluidsynth_version(),
            "score": str(score_path),
            "performance": str(performance_path),
            "soundfont_sha256": soundfont_hash,
            "cases": [f"{method}_{processor}" for method in METHODS for processor in PROCESSORS],
        },
    )


if __name__ == "__main__":
    main()
