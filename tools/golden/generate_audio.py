"""Generate audio fixtures and reference outputs for matchmaker-rs.

Run from ``tools/golden`` with the locked environment. The reference renderer
uses Partitura/FluidSynth with the pinned local SF2; the Rust runtime remains
pure Rust and uses RustySynth.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
from ctypes import c_int
from contextlib import contextmanager, redirect_stdout
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import librosa
import matchmaker
import numpy as np
import partitura
import soundfile as sf
from matchmaker import Matchmaker
from matchmaker.features.audio import (
    SAMPLE_RATE,
    ChromagramProcessor,
    CQTProcessor,
    LogSpectralEnergyProcessor,
    RawSpectrumProcessor,
)
from matchmaker.dp.oltw_arzt import OnlineTimeWarpingArztFrame
from matchmaker.matchmaker import DEFAULT_TEMPO
from matchmaker.prob import skf as reference_skf
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


def stream_raw_spectrum(samples: np.ndarray, sample_rate: int, hop: int, n_fft: int):
    """Features and timestamps the reference stream adapter yields for ``samples``."""
    processor = RawSpectrumProcessor(sample_rate, hop, n_fft)
    padded = np.pad(samples, (0, (-len(samples)) % hop), mode="constant")
    cache = np.zeros(n_fft - hop, dtype=np.float32)
    feature_rows, feature_times = [], []
    for chunk_index, chunk in enumerate(padded.reshape(-1, hop)):
        block = np.concatenate((cache, chunk))
        features, frame_time = processor((block, max(0, chunk_index - 1) * hop / sample_rate))
        if chunk_index > 0:
            feature_rows.extend(features)
            feature_times.extend([float(frame_time)] * len(features))
        cache = block[-len(cache):].copy()
    return feature_rows, feature_times


def generate_raw_spectrum_goldens(output: Path) -> None:
    """Generate default, override, short, and silent raw-spectrum references."""
    base = output / "processors/raw_spectrum"
    configurations = (
        ("default", 8_000, 128, 512),
        ("override", 16_000, 256, 1_024),
    )

    def write_case(name: str, samples: np.ndarray, sample_rate: int, hop: int, n_fft: int) -> None:
        feature_rows, feature_times = stream_raw_spectrum(samples, sample_rate, hop, n_fft)

        directory = base / name
        write_array(directory / "input_samples.npy", samples, "<f4")
        write_array(
            directory / "features.npy",
            np.asarray(feature_rows, dtype=np.float32).reshape(-1, n_fft // 2 + 1),
            "<f4",
        )
        write_array(directory / "times.npy", feature_times, "<f8")
        write_json(
            directory / "meta.json",
            {
                "reference": reference_info(),
                "generator": "tools/golden/generate_audio.py",
                "processor": "raw_spectrum",
                "sample_rate": sample_rate,
                "hop_length": hop,
                "n_fft": n_fft,
                "center": False,
                "window": "hann",
                "frames": len(feature_rows),
                "tolerances": {"features_rtol": 1e-5, "features_atol": 1e-5},
            },
        )
        print(f"raw_spectrum/{name}: {len(feature_rows)} frames")

    for name, sample_rate, hop, n_fft in configurations:
        times = np.arange(sample_rate // 2, dtype=np.float64) / sample_rate
        samples = (
            0.35 * np.sin(2 * np.pi * 440.0 * times)
            + 0.2 * np.sin(2 * np.pi * 659.2551138257398 * times)
            + 0.1 * np.sin(2 * np.pi * 783.9908719634985 * times)
        ).astype(np.float32)
        write_case(name, samples, sample_rate, hop, n_fft)

    silent = np.zeros(2_048, dtype=np.float32)

    # shortcut: 4 MB and read by no test; options in research.md "C04 - Open improvement".
    performance_samples, performance_rate = sf.read(
        DEFAULT_PERFORMANCE, dtype=np.float32, always_2d=True
    )
    performance_samples = performance_samples.mean(axis=1, dtype=np.float32)
    write_case("performance", performance_samples, performance_rate, 128, 512)
    write_case("silent", silent, 8_000, 128, 512)
    short = np.zeros(511, dtype=np.float32)
    short_directory = base / "short"
    write_array(short_directory / "input_samples.npy", short, "<f4")
    write_array(short_directory / "features.npy", np.empty((0, 257)), "<f4")
    write_array(short_directory / "times.npy", [], "<f8")
    write_json(
        short_directory / "meta.json",
        {
            "reference": reference_info(),
            "generator": "tools/golden/generate_audio.py",
            "processor": "raw_spectrum",
            "sample_rate": 8_000,
            "hop_length": 128,
            "n_fft": 512,
            "center": False,
            "window": "hann",
            "frames": 0,
            "direct_call": "librosa raises when n_fft exceeds uncentered input length",
        },
    )
    print("raw_spectrum/short: 0 frames (reference direct call rejects short input)")


def stream_cqt(samples: np.ndarray, sample_rate: int, hop: int, norm=np.inf):
    """Rows and timestamps the reference stream adapter yields for ``samples``.

    ``CQTProcessor`` has no ``n_fft``, so the adapter caches one hop: every call sees two hops.
    """
    processor = CQTProcessor(sample_rate, hop, norm)
    padded = np.pad(samples, (0, (-len(samples)) % hop), mode="constant")
    cache = np.zeros(hop, dtype=np.float32)
    rows, times = [], []
    for chunk_index, chunk in enumerate(padded.reshape(-1, hop)):
        block = np.concatenate((cache, chunk))
        features, frame_time = processor((block, max(0, chunk_index - 1) * hop / sample_rate))
        if chunk_index > 0:
            rows.extend(features)
            times.extend([float(frame_time)] * len(features))
        cache = block[-hop:].copy()
    return rows, times


@contextmanager
def record_cqt_stages():
    """Capture the soxr outputs and sparse kernels of the reference ``librosa.cqt`` calls."""
    from librosa.core import audio as librosa_audio
    from librosa.core import constantq

    stages = {"resample": [], "kernels": []}
    original_resample = librosa_audio.resample
    original_kernel = getattr(constantq, "__vqt_filter_fft")

    def resample(y, *, orig_sr, target_sr, **kwargs):
        out = original_resample(y, orig_sr=orig_sr, target_sr=target_sr, **kwargs)
        stages["resample"].append(
            {"factor": int(orig_sr // target_sr), "input_length": int(len(y)), "output": np.array(out, dtype=np.float32)}
        )
        return out

    def kernel(sample_rate, *args, **kwargs):
        basis, n_fft, lengths = original_kernel(sample_rate, *args, **kwargs)
        csr = basis.tocsr()  # recorded before librosa rescales it for the downsampled rate
        stages["kernels"].append(
            {
                "sample_rate": float(sample_rate),
                "n_fft": int(n_fft),
                "indptr": csr.indptr.copy(),
                "indices": csr.indices.copy(),
                "data": csr.data.copy(),
            }
        )
        return basis, n_fft, lengths

    librosa_audio.resample = resample
    setattr(constantq, "__vqt_filter_fft", kernel)
    try:
        yield stages
    finally:
        librosa_audio.resample = original_resample
        setattr(constantq, "__vqt_filter_fft", original_kernel)


def generate_cqt_goldens(output: Path) -> None:
    """Stream rows, whole-signal rows, soxr decimations and kernels of ``CQTProcessor``."""
    base = output / "processors/cqt"
    performance, performance_rate = sf.read(DEFAULT_PERFORMANCE, dtype=np.float32, always_2d=True)
    performance = performance.mean(axis=1, dtype=np.float32)
    assert performance_rate == 44_100
    # Amended for cqt only (research R13): the reference's float32/soxr noise is amplified in the
    # low bins, so a per-element limit also gets a frame-maximum (kernel: row-maximum) term.
    tolerances = {
        "features_rtol": 1e-5,
        "features_atol": 1e-5,
        "features_frame_max_rtol": 1e-5,
        "decimated_rtol": 1e-5,
        "decimated_atol": 1e-5,
        "kernel_rtol": 1e-5,
        "kernel_atol": 1e-5,
        "kernel_row_max_rtol": 1e-5,
    }

    def tones(sample_rate: int, seconds: float, low: bool = False) -> np.ndarray:
        times = np.arange(int(sample_rate * seconds), dtype=np.float64) / sample_rate
        signal = (
            0.35 * np.sin(2 * np.pi * 440.0 * times)
            + 0.2 * np.sin(2 * np.pi * 659.2551138257398 * times)
            + 0.1 * np.sin(2 * np.pi * 783.9908719634985 * times)
        )
        if low:
            signal += 0.25 * np.sin(2 * np.pi * 110.0 * times)
        return signal.astype(np.float32)

    def write_case(name, samples, sample_rate, hop, whole_samples, norm=np.inf, what=""):
        rows, times = stream_cqt(samples, sample_rate, hop, norm)
        processor = CQTProcessor(sample_rate, hop, norm)
        with record_cqt_stages() as stages:
            whole, _ = processor((samples[:whole_samples], 0.0))
        directory = base / name
        write_array(directory / "input_samples.npy", samples, "<f4")
        write_array(directory / "stream_features.npy", np.asarray(rows, dtype=np.float32).reshape(-1, 88), "<f4")
        write_array(directory / "stream_times.npy", times, "<f8")
        write_array(directory / "whole_features.npy", whole, "<f4")
        for index, stage in enumerate(stages["resample"]):
            write_array(directory / f"decimated_{index}.npy", stage["output"], "<f4")
        kernels = stages["kernels"]
        write_array(directory / "kernel_octave_rows.npy", [len(k["indptr"]) - 1 for k in kernels], "<i4")
        write_array(directory / "kernel_octave_n_fft.npy", [k["n_fft"] for k in kernels], "<i4")
        write_array(directory / "kernel_octave_rate.npy", [k["sample_rate"] for k in kernels], "<f8")
        write_array(directory / "kernel_row_nnz.npy", np.concatenate([np.diff(k["indptr"]) for k in kernels]), "<i4")
        write_array(directory / "kernel_indices.npy", np.concatenate([k["indices"] for k in kernels]), "<i4")
        write_array(
            directory / "kernel_values.npy",
            np.concatenate([np.stack([k["data"].real, k["data"].imag], axis=1) for k in kernels]),
            "<f4",
        )
        write_json(
            directory / "meta.json",
            {
                "reference": reference_info(),
                "generator": "tools/golden/generate_audio.py",
                "processor": "cqt",
                "description": what,
                "sample_rate": sample_rate,
                "hop_length": hop,
                "norm": "inf" if norm == np.inf else norm,
                "n_bins": 88,
                "fmin_hz": float(librosa.note_to_hz("A0")),
                "whole_samples": int(whole_samples),
                "stream_frames": len(rows),
                "whole_frames": int(len(whole)),
                "resamples": [
                    {"factor": s["factor"], "input_length": s["input_length"], "output_length": len(s["output"])}
                    for s in stages["resample"]
                ],
                "librosa": librosa.__version__,
                "soxr": __import__("soxr").__libsoxr_version__,
                "tolerances": tolerances,
            },
        )
        print(f"cqt/{name}: {len(rows)} stream rows, {len(whole)} whole rows, {len(kernels)} octaves, {len(stages['resample'])} decimations")

    start = 22_050
    write_case("default", performance[start : start + 35_280], 44_100, 1_470, 22_050, what="piano, 44.1 kHz, one decimation")
    at_16k = librosa.resample(performance[start : start + 44_100], orig_sr=44_100, target_sr=16_000, res_type="soxr_hq")
    write_case("override", at_16k[:12_800].astype(np.float32), 16_000, 640, 8_000, what="piano, 16 kHz, seven decimations")
    write_case("tones", tones(44_100, 0.4), 44_100, 735, 17_640, what="odd hop: no decimation at all")
    write_case("early", tones(44_100, 0.4, low=True), 44_100, 2_048, 17_640, what="hop 2048: early downsampling by 2")
    write_case("norm2", tones(44_100, 0.25), 44_100, 735, 11_025, norm=2.0, what="filter norm 2 instead of infinity")
    write_case("silent", np.zeros(4_410, dtype=np.float32), 44_100, 1_470, 4_410, what="digital silence")
    short = (0.1 * np.sin(np.arange(100) * 0.3)).astype(np.float32)
    write_case("short", short, 44_100, 1_470, 100, what="fewer samples than one hop: no row survives the edge trim")

    # The whole bundled performance, rows only: `matchmaker features` exports of the same file are
    # compared with these (tools/golden/compare.py); the WAV itself is the input.
    rows, times = stream_cqt(performance, 44_100, 1_470)
    directory = base / "performance"
    write_array(directory / "stream_features.npy", np.asarray(rows, dtype=np.float32).reshape(-1, 88), "<f4")
    write_array(directory / "stream_times.npy", times, "<f8")
    write_json(
        directory / "meta.json",
        {
            "reference": reference_info(),
            "generator": "tools/golden/generate_audio.py",
            "processor": "cqt",
            "description": "bundled performance WAV, whole file; no input copy",
            "input": "tests/data/audio/simple_mozart_k265_var1_sf2.wav",
            "sample_rate": 44_100,
            "hop_length": 1_470,
            "norm": "inf",
            "n_bins": 88,
            "stream_frames": len(rows),
            "librosa": librosa.__version__,
            "soxr": __import__("soxr").__libsoxr_version__,
            "tolerances": tolerances,
        },
    )
    print(f"cqt/performance: {len(rows)} stream rows")


# Frames whose full hypothesis state is stored for the real-score replay.
SKF_STATE_FRAMES = (0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89)


def write_skf_case(
    directory: Path,
    note_array: np.ndarray,
    tempo: float,
    templates: np.ndarray,
    spectra: np.ndarray,
    follower: dict,
    state_frames,
    meta: dict,
    store_notes: bool = True,
    template_dtype: str = "<f8",
) -> None:
    """Replay the reference SKF with fixed templates and save inputs, positions and state.

    The reference builds its templates in the constructor, so the fixed ones are injected by
    replacing ``build_spectral_templates``; ``templates`` is exactly what both sides consume.
    """
    times = np.arange(len(spectra)) * follower["hop_length"] / follower["sample_rate"]
    with patch.object(reference_skf, "build_spectral_templates", lambda chords, **_: templates):
        with redirect_stdout(io.StringIO()):
            reference = reference_skf.SwitchingKalmanFilterFollower(
                note_array, queue=None, tempo=tempo, **follower
            )
    chords, lengths, onset_beats = reference_skf.build_chord_sequence(note_array)
    positions, current, states = [], [], []
    for index, (spectrum, time) in enumerate(zip(spectra, times)):
        positions.append(reference(spectrum, float(time)))
        current.append(reference.current_index)
        if index in state_frames:
            states.append((index, list(reference.hypotheses.items())))

    if store_notes:
        write_array(directory / "note_onset_beat.npy", note_array["onset_beat"], "<f4")
        write_array(directory / "note_onset_quarter.npy", note_array["onset_quarter"], "<f4")
        write_array(directory / "note_pitch.npy", note_array["pitch"], "<i4")
    write_array(directory / "chords_flat.npy", [pitch for chord in chords for pitch in chord], "<i4")
    write_array(directory / "chord_offsets.npy", np.cumsum([0] + [len(chord) for chord in chords]), "<i8")
    write_array(directory / "lengths.npy", lengths, "<f8")
    write_array(directory / "onset_beats.npy", onset_beats, "<f8")
    write_array(directory / "templates.npy", templates, template_dtype)
    write_array(directory / "observations.npy", spectra, "<f4")
    write_array(directory / "perf_times.npy", times, "<f8")
    write_array(directory / "positions.npy", positions, "<f8")
    write_array(directory / "current_index.npy", current, "<i8")
    offsets, chord_ids, ages, probabilities, means, variances = [0], [], [], [], [], []
    for _, items in states:
        for (chord, age), (probability, mean, variance) in items:
            chord_ids.append(chord)
            ages.append(age)
            probabilities.append(probability)
            means.append(mean)
            variances.append(variance)
        offsets.append(len(chord_ids))
    write_array(directory / "state_frames.npy", [index for index, _ in states], "<i8")
    write_array(directory / "state_offsets.npy", offsets, "<i8")
    write_array(directory / "state_chord.npy", chord_ids, "<i4")
    write_array(directory / "state_age.npy", ages, "<i4")
    write_array(directory / "state_probability.npy", probabilities, "<f8")
    write_array(directory / "state_tempo_mean.npy", means, "<f8")
    write_array(directory / "state_tempo_variance.npy", variances, "<f8")
    write_json(
        directory / "meta.json",
        {
            "reference": reference_info(),
            "generator": "tools/golden/generate_audio.py",
            "method": "skf",
            "tempo": tempo,
            "follower": follower,
            "frames": len(spectra),
            "chords": len(chords),
            "tolerances": {"model_rtol": 1e-7, "model_atol": 1e-9, "position_atol": 1e-9},
            **meta,
        },
    )
    print(
        f"skf/{directory.name}: {len(chords)} chords, {len(spectra)} frames, "
        f"max {max(len(items) for _, items in states)} hypotheses in stored states"
    )


def generate_skf_goldens(output: Path, soundfont: Path, soundfont_hash: str) -> None:
    """Fixed-template SKF replays: synthetic Gaussian cases and a real score with synthesized templates."""
    from matchmaker.utils.misc import get_tempo_from_score

    base = output / "skf"
    rng = np.random.default_rng(1984)
    note_dtype = [("onset_beat", "<f4"), ("onset_quarter", "<f4"), ("pitch", "<i4")]

    def synthetic(name, quarters, pitches, beat_scale, durations, tempo, follower, specials=()):
        """Gaussian-template case: score table (shuffled rows), spectra that follow the chords."""
        rows = [(q * beat_scale, q, p) for q, chord in zip(quarters, pitches) for p in chord]
        table = np.array(rows, dtype=note_dtype)[rng.permutation(len(rows))]
        chords, _, _ = reference_skf.build_chord_sequence(table)
        n_bins = follower["n_fft"] // 2 + 1
        templates = reference_skf._build_gaussian_templates(
            chords, follower["n_fft"], follower["sample_rate"], 5
        )
        spectra = []
        for chord, frames in enumerate(durations):
            for _ in range(frames):
                spectra.append(
                    templates[chord] * (0.8 + 0.4 * rng.random()) + 5e-4 * rng.random(n_bins)
                )
        spectra = np.asarray(spectra, dtype=np.float32)
        for index, kind in specials:
            spectra[index] = 0.0 if kind == "silent" else spectra[index] - spectra[index].mean()
        write_skf_case(
            base / name,
            table,
            tempo,
            templates,
            spectra,
            follower,
            range(len(spectra)),
            {
                "template_mode": "gaussian",
                "input": "synthetic spectra drawn around the Gaussian templates (numpy PCG64 seed 1984)",
                "special_frames": [[index, kind] for index, kind in specials],
            },
        )

    synthetic(
        "synthetic",
        [0.0, 1.0, 2.0, 2.5, 3.0, 4.0, 5.5, 6.5],
        [[60], [64, 67], [62], [65, 69, 72], [60, 64], [67], [71, 74], [72]],
        2.0,
        [11, 10, 6, 5, 12, 16, 10, 11],
        240.0,
        {"sample_rate": 8000, "n_fft": 512, "hop_length": 128, "max_hypotheses": 12},
        specials=((35, "silent"), (50, "negative")),
    )
    synthetic(
        "override",
        [0.0, 1.0, 2.0],
        [[69], [72, 76], [64]],
        1.0,
        [6, 8, 9],
        240.0,
        {"sample_rate": 16000, "n_fft": 1024, "hop_length": 512},
    )
    synthetic(
        "one_chord",
        [0.0],
        [[60, 64, 67]],
        1.0,
        [12],
        100.0,
        {"sample_rate": 8000, "n_fft": 512, "hop_length": 128},
    )

    score_path = DEFAULT_SCORE
    _, score_part, _ = load_score_like_matchmaker(score_path)
    note_array = score_part.note_array()
    score_tempo = get_tempo_from_score(score_part, str(score_path))
    tempo = float(score_tempo) if score_tempo is not None else float(DEFAULT_TEMPO)
    chords, _, _ = reference_skf.build_chord_sequence(note_array)
    # Pin the mode: the reference silently falls back to Gaussian peaks on any synthesis error.
    printed = io.StringIO()
    with use_soundfont(soundfont), redirect_stdout(printed):
        synthesized = reference_skf.build_spectral_templates(chords, n_fft=512, sample_rate=8000)
    if "synthesized" not in printed.getvalue():
        raise RuntimeError(f"reference did not synthesize its templates: {printed.getvalue()!r}")
    templates = synthesized.astype(np.float32).astype(np.float64)
    audio, rate = librosa.load(str(DEFAULT_PERFORMANCE), sr=None)
    audio = librosa.resample(y=audio, orig_sr=rate, target_sr=8000).astype(np.float32)
    rows, _ = stream_raw_spectrum(audio, 8000, 128, 512)
    spectra = np.asarray(rows[: SKF_STATE_FRAMES[-1] + 1], dtype=np.float32)
    write_skf_case(
        base / "bundled",
        note_array,
        tempo,
        templates,
        spectra,
        {"sample_rate": 8000, "n_fft": 512, "hop_length": 128},
        SKF_STATE_FRAMES,
        {
            "template_mode": "synthesized",
            "template_note": "reference templates rounded to float32; both sides replay these values",
            "score": score_path.name,
            "performance": DEFAULT_PERFORMANCE.name,
            "performance_sha256": sha256(DEFAULT_PERFORMANCE),
            "soundfont_sha256": soundfont_hash,
            "fluidsynth": fluidsynth_version(),
        },
        store_notes=False,
        template_dtype="<f4",
    )



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


def multi_fold_parts(score_part):
    clones = []
    try:
        clones.append(
            partitura.score.unfold_part_maximal(score_part, ignore_leaps=False)
        )
    except Exception:
        pass
    clones.append(score_part)
    try:
        clones.extend(itertools.islice(
            partitura.score.iter_unfolded_parts(score_part, update_ids=True), 4
        ))
    except Exception:
        pass

    unique = []
    seen = set()
    for clone in clones:
        notes = clone.note_array()
        if not len(notes):
            continue
        key = (len(notes), round(float(notes["onset_beat"].max()), 2))
        if key not in seen:
            seen.add(key)
            unique.append(clone)
    return unique or [score_part]


def generate_multi_fold_candidates(score_path: Path, output: Path) -> None:
    """Record Partitura's repeat-path order without rendering or third-party audio."""
    from partitura.score import merge_parts

    loaded = partitura.load_musicxml(
        str(score_path), ignore_invisible_objects=True, force_note_ids="keep"
    )
    folded = merge_parts(loaded.parts)
    maximal_score = partitura.score.unfold_part_maximal(
        loaded, ignore_leaps=False
    )
    maximal = merge_parts(maximal_score.parts)
    candidate_sets = {
        "folded_input": multi_fold_parts(folded),
        "maximally_unfolded_input": multi_fold_parts(maximal),
    }
    base = output / "multi_fold" / score_path.stem
    metadata = {
        "reference": reference_info(),
        "generator": "tools/golden/generate_audio.py",
        "score": score_path.name,
        "score_sha256": sha256(score_path),
        "input": "synthetic hand-authored repeat/volta MusicXML fixture",
        "selection_window": 50,
        "switch_margin": 0.05,
        "deduplication": ["note_count", "round(max_onset_beat, 2)"],
        "candidate_sets": {},
    }
    for set_name, candidates in candidate_sets.items():
        entries = []
        for index, clone in enumerate(candidates):
            notes = clone.note_array()
            note_values = np.column_stack((
                notes["pitch"], notes["onset_beat"], notes["onset_quarter"]
            )).astype(np.float64)
            write_array(base / set_name / f"candidate_{index}_notes.npy", note_values, "<f8")
            entries.append({
                "index": index,
                "note_count": len(notes),
                "max_onset_beat": float(notes["onset_beat"].max()),
                "note_ids": notes["id"].astype(str).tolist(),
            })
        metadata["candidate_sets"][set_name] = entries
    write_json(base / "meta.json", metadata)
    print("arzt_multi_fold/candidates: " + ", ".join(
        f"{name}={len(candidates)}" for name, candidates in candidate_sets.items()
    ))


def generate_multi_fold_audio(
    score_path: Path, output: Path, soundfont: Path, soundfont_hash: str
) -> None:
    """Generate taken/skipped-repeat audio and locked-reference follower traces."""
    from partitura.score import merge_parts

    loaded = partitura.load_musicxml(
        str(score_path), ignore_invisible_objects=True, force_note_ids="keep"
    )
    folded = merge_parts(loaded.parts)
    candidates = multi_fold_parts(folded)
    tempo = reference_misc.get_tempo_from_score(folded, str(score_path))
    tempo = float(tempo) if tempo is not None else float(DEFAULT_TEMPO)
    audio_dir = ROOT / "tests/data/audio/arzt_multi_fold"
    audio_dir.mkdir(parents=True, exist_ok=True)

    for name, part in (("repeat_taken", candidates[0]), ("repeat_skipped", candidates[-1])):
        performance = audio_dir / f"{name}.wav"
        samples = render_with_soundfont(part, tempo, soundfont)
        wavfile.write(str(performance), SAMPLE_RATE, samples)

        recorder = {
            "features": [],
            "times": [],
            "positions": [],
            "selected": [],
            "means": [],
            "frames": [],
            "local_costs": [],
        }
        kwargs = {
            "processor": "chroma",
            "sample_rate": SAMPLE_RATE,
            "frame_rate": 30,
            "window_size": 10,
            "start_window_size": 0.1,
            "step_size": 3,
            "selection_window": 50,
            "switch_margin": 0.05,
        }
        with use_soundfont(soundfont):
            mm = Matchmaker(
                score_file=score_path,
                performance_file=performance,
                input_type="audio",
                method="arzt_multi_fold",
                processor="chroma",
                kwargs=kwargs,
                unfold_score=False,
            )
            record_follower(mm.score_follower, recorder)
            yielded = list(mm.run(verbose=False))
        if not np.array_equal(np.asarray(yielded), np.asarray(recorder["positions"])):
            raise RuntimeError("Recorded multi-fold positions differ from yielded positions")

        folded_note_beats = {
            str(note["id"]): float(note["onset_beat"])
            for note in folded.note_array()
        }
        note_events = set()
        for note in part.note_array():
            note_id = str(note["id"])
            prefix, separator, suffix = note_id.rpartition("-")
            original_id = (
                prefix if separator and suffix.isdigit() else note_id
            )
            if original_id in folded_note_beats:
                note_events.add((
                    float(note["onset_quarter"]) * 60.0 / tempo,
                    folded_note_beats[original_id],
                ))
        truth = sorted(note_events)
        truth_path = audio_dir / f"{name}.tsv"
        truth_path.write_text(
            "perf_sec\tscore_beat\n"
            + "".join(
                f"{time:.17g}\t{beat:.17g}\n" for time, beat in truth
            ),
            encoding="utf-8",
        )

        base = output / "multi_fold" / name
        (base / "ground_truth.tsv").write_text(
            truth_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        write_array(base / "performance_features.npy", recorder["features"], "<f4")
        write_array(base / "performance_times.npy", recorder["times"], "<f8")
        write_array(base / "positions.npy", recorder["positions"], "<f8")
        write_array(base / "selected_hypothesis.npy", recorder["selected"], "<i8")
        write_array(base / "candidate_mean_costs.npy", recorder["means"], "<f8")
        write_array(base / "candidate_frames.npy", recorder["frames"], "<i8")
        write_array(base / "candidate_local_costs.npy", recorder["local_costs"], "<f8")
        write_array(
            base / "alignment_path.npy",
            np.asarray(mm.score_follower.alignment_path, dtype=np.float64).reshape(2, -1),
            "<f8",
        )
        hypotheses = []
        for index, hypothesis in enumerate(mm.score_follower.hypotheses):
            candidate_dir = base / f"candidate_{index}"
            tracker = hypothesis["tracker"]
            clone_notes = hypothesis["clone"].note_array()
            write_array(candidate_dir / "score_features.npy", tracker.reference_features, "<f4")
            write_array(candidate_dir / "score_positions.npy", tracker.score_positions, "<f4")
            write_array(candidate_dir / "frame_to_unfolded_beat.npy", tracker._ref_frame_to_beat, "<f8")
            write_array(candidate_dir / "frame_to_notated_beat.npy", hypothesis["ref_frame_to_notated_beat"], "<f4")
            note_values = np.column_stack((
                clone_notes["pitch"], clone_notes["onset_beat"], clone_notes["onset_quarter"]
            )).astype(np.float64)
            write_array(candidate_dir / "notes.npy", note_values, "<f8")
            hypotheses.append({
                "index": index,
                "note_count": len(clone_notes),
                "max_onset_beat": float(clone_notes["onset_beat"].max()),
                "reference_frames": len(tracker.reference_features),
            })
        write_json(
            base / "meta.json",
            {
                "reference": reference_info(),
                "fluidsynth": fluidsynth_version(),
                "generator": "tools/golden/generate_audio.py",
                "score": score_path.name,
                "score_sha256": sha256(score_path),
                "performance": performance.name,
                "performance_sha256": sha256(performance),
                "ground_truth": truth_path.name,
                "soundfont_sha256": soundfont_hash,
                "tempo": tempo,
                "sample_rate": SAMPLE_RATE,
                "frame_rate": 30,
                "processor": "chroma",
                "method": "arzt_multi_fold",
                "unfold_score": False,
                "selection_window": 50,
                "switch_margin": 0.05,
                "hypotheses": hypotheses,
            },
        )
        print(f"arzt_multi_fold/{name}: {len(recorder['positions'])} frames")


def generate_audio_ground_truth(
    score_path: Path,
    performance_path: Path,
    output: Path,
    soundfont_hash: str,
    soundfont: Path,
) -> None:
    """Record direct score-onset TSV ground truth and reference audio metrics."""
    from matchmaker.utils.eval import evaluate_alignment

    kwargs = {
        "processor": "chroma",
        "sample_rate": SAMPLE_RATE,
        **METHODS["arzt"],
    }
    with use_soundfont(soundfont):
        mm = Matchmaker(
            score_file=score_path,
            performance_file=performance_path,
            input_type="audio",
            method="arzt",
            processor="chroma",
            kwargs=kwargs,
        )
        list(mm.run(verbose=False))

    note_beats = np.unique(mm.score_part.note_array()["onset_beat"]).astype(float)
    frame_mapping = mm.ref_frame_to_beat().astype(float)
    frame_indices = np.interp(
        note_beats, frame_mapping, np.arange(len(frame_mapping), dtype=float)
    )
    perf_times = frame_indices / float(mm.frame_rate)
    if len(note_beats) > 8:
        selected = np.linspace(0, len(note_beats) - 1, 8).round().astype(int)
        note_beats, perf_times = note_beats[selected], perf_times[selected]

    directory = ROOT / "tests/data/audio/ground_truth"
    directory.mkdir(parents=True, exist_ok=True)
    tsv = directory / "score_onsets.tsv"
    tsv.write_text(
        "perf_sec\tscore_beat\n"
        + "".join(
            f"{time:.17g}\t{beat:.17g}\n"
            for time, beat in zip(perf_times, note_beats)
        ),
        encoding="utf-8",
    )
    path = np.asarray(mm.score_follower.alignment_path, dtype=np.float64)
    metrics = evaluate_alignment(path[1], path[0], note_beats, perf_times)
    base = output / "ground_truth"
    write_array(base / "alignment_path.npy", path, "<f8")
    write_array(base / "ground_truth.npy", np.column_stack((perf_times, note_beats)), "<f8")
    (base / "ground_truth.tsv").write_text(tsv.read_text(encoding="utf-8"), encoding="utf-8")
    write_json(base / "evaluation.json", metrics)
    write_json(
        base / "meta.json",
        {
            "reference": reference_info(),
            "generator": "tools/golden/generate_audio.py",
            "score": score_path.name,
            "score_sha256": sha256(score_path),
            "performance": performance_path.name,
            "performance_sha256": sha256(performance_path),
            "soundfont_sha256": soundfont_hash,
            "method": "audio_ground_truth",
            "processor": "chroma",
            "ground_truth": "sampled unique score onsets mapped to performance seconds",
            "ground_truth_file": tsv.name,
            "events": len(note_beats),
            "metric_source": "matchmaker.utils.eval.evaluate_alignment",
        },
    )
    print(f"audio/ground_truth: {len(note_beats)} note onsets")


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
        if "selected" in recorder:
            recorder["selected"].append(self.active_hypothesis_idx)
            recorder["means"].append([
                hyp["mean_recent_cost"] for hyp in self.hypotheses
            ])
            frames, costs = [], []
            for hypothesis in self.hypotheses:
                tracker = hypothesis["tracker"]
                frame = tracker._current_frame
                frames.append(frame)
                costs.append(float(tracker.distance_func(
                    tracker.reference_features[frame],
                    np.asarray(observation, dtype=np.float32).squeeze(),
                )))
            recorder["frames"].append(frames)
            recorder["local_costs"].append(costs)
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
        "--raw-spectrum-only",
        action="store_true",
        help="generate raw-spectrum processor goldens without FluidSynth",
    )
    parser.add_argument(
        "--cqt-only",
        action="store_true",
        help="generate cqt processor goldens (stream, whole-signal, soxr stages, kernels)",
    )
    parser.add_argument(
        "--skf-only",
        action="store_true",
        help="generate fixed-template SKF replays (needs FluidSynth and the pinned SoundFont)",
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
    parser.add_argument(
        "--multi-fold-candidates-only",
        action="store_true",
        help="generate renderer-free arzt_multi_fold repeat-path goldens",
    )
    parser.add_argument(
        "--multi-fold-only",
        action="store_true",
        help="generate taken/skipped-repeat arzt_multi_fold audio and follower goldens",
    )
    parser.add_argument(
        "--audio-ground-truth-only",
        action="store_true",
        help="generate a direct audio score-onset TSV and metric golden",
    )
    args = parser.parse_args()
    if args.tempo_replay_only:
        generate_tempo_replay_goldens(args.out.resolve())
        return
    if args.multi_fold_candidates_only:
        score_path = args.score.resolve()
        if args.score == DEFAULT_SCORE:
            score_path = ROOT / "tests/data/fixture_repeats_voltas.musicxml"
        if not score_path.is_file():
            parser.error(f"score does not exist: {score_path}")
        generate_multi_fold_candidates(score_path, args.out.resolve())
        return
    if args.multi_fold_only:
        score_path = args.score.resolve()
        if args.score == DEFAULT_SCORE:
            score_path = ROOT / "tests/data/fixture_repeats_voltas.musicxml"
        soundfont = args.soundfont.resolve()
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
        generate_multi_fold_candidates(score_path, args.out.resolve())
        generate_multi_fold_audio(score_path, args.out.resolve(), soundfont, soundfont_hash)
        return
    if args.audio_ground_truth_only:
        score_path = args.score.resolve()
        performance_path = args.performance.resolve()
        soundfont = args.soundfont.resolve()
        if not score_path.is_file():
            parser.error(f"score does not exist: {score_path}")
        if not performance_path.is_file():
            parser.error(f"performance does not exist: {performance_path}")
        if not soundfont.is_file():
            parser.error(f"SoundFont does not exist: {soundfont}")
        soundfont_hash = sha256(soundfont)
        if soundfont_hash != SOUNDFONT_SHA256:
            parser.error(
                f"unexpected SoundFont SHA-256 {soundfont_hash}; "
                f"expected pinned asset {SOUNDFONT_SHA256}"
            )
        generate_audio_ground_truth(
            score_path,
            performance_path,
            args.out.resolve(),
            soundfont_hash,
            soundfont,
        )
        return
    if args.codec_fixtures_only:
        generate_codec_fixtures(ROOT / "tests/data/audio/codec_samples")
        return
    if args.processors_only:
        generate_processor_goldens(args.out.resolve())
        return
    if args.raw_spectrum_only:
        generate_raw_spectrum_goldens(args.out.resolve())
        return
    if args.cqt_only:
        generate_cqt_goldens(args.out.resolve())
        return
    if args.skf_only:
        soundfont = args.soundfont.resolve()
        if not soundfont.is_file():
            parser.error(f"SoundFont does not exist: {soundfont}")
        soundfont_hash = sha256(soundfont)
        if soundfont_hash != SOUNDFONT_SHA256:
            parser.error(
                f"unexpected SoundFont SHA-256 {soundfont_hash}; "
                f"expected pinned asset {SOUNDFONT_SHA256}"
            )
        generate_skf_goldens(args.out.resolve(), soundfont, soundfont_hash)
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
