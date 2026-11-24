#!/usr/bin/env python3
"""
RAVEGRID stem-separator with optional specialist vocal/bass upgrades.

Usage:
    python stem_separator.py /path/to/audio.mp3
    python stem_separator.py /path/to/audio.wav --model htdemucs_ft
    python stem_separator.py /path/to/audio.mp3 --vocal-model model_bs_roformer_ep_317_sdr_12.9755.ckpt
    python stem_separator.py /path/to/audio.mp3 --bass-model kuielab_a_bass.onnx
    python stem_separator.py /path/to/audio.mp3 --list-models bass

Outputs are written to:
    <current-working-directory>/<input-name>_stems/

Stem outputs:
    drums  -> drumkit.wav
    bass   -> bass.wav       (upgraded if --bass-model is provided)
    other  -> synth.wav      (the residual "other" source, not synth-only)
    vocals -> vocal.wav      (upgraded if --vocal-model is provided; defaults to None for speed)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
STEM_MAPPING = {
    "drums": "drumkit",
    "bass": "bass",
    "other": "synth",
    "vocals": "vocal",
}
DEFAULT_SPECIALIST_MODEL_DIR = Path.home() / ".cache" / "ravegrid" / "audio-separator-models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Separate audio into drumkit, bass, synth/other, and vocal stems, "
        "with optional specialist models for cleaner vocals and bass."
    )
    parser.add_argument("input", nargs="?", type=Path, help="Path to the input audio file.")
    parser.add_argument(
        "--list-models",
        choices=("vocals", "bass"),
        help="List candidate audio-separator models for a stem and exit.",
    )
    parser.add_argument(
        "--model",
        choices=("htdemucs", "htdemucs_ft"),
        default="htdemucs",
        help="Demucs baseline model (default: htdemucs).",
    )
    parser.add_argument(
        "--vocal-model",
        type=str,
        default=None,
        help="Optional audio-separator vocal model filename to replace/upgrade vocal.wav "
        "(e.g. model_bs_roformer_ep_317_sdr_12.9755.ckpt).",
    )
    parser.add_argument(
        "--bass-model",
        type=str,
        default=None,
        help="Optional audio-separator bass model filename to replace/upgrade bass.wav.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_SPECIALIST_MODEL_DIR,
        help=f"Directory for cached specialist models (default: {DEFAULT_SPECIALIST_MODEL_DIR}).",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Inference device (default: cpu).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=3,
        help="Maximum PyTorch CPU threads (default: 3).",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.25,
        help="Overlap between Demucs segments (default: 0.25).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Default: ./<input-name>_stems",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    return parser.parse_args()


def list_specialist_models(stem: str) -> int:
    executable = shutil.which("audio-separator")
    if not executable:
        print(
            "Could not find 'audio-separator' on PATH. Install audio-separator to use specialist models.",
            file=sys.stderr,
        )
        return 2

    command = [
        executable,
        "--list_models",
        f"--list_filter={stem}",
        "--list_limit=15",
    ]
    return subprocess.run(command, check=False).returncode


def validate_args(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.input is None:
        raise ValueError("Provide an input audio file, or pass --list-models {vocals,bass}.")

    input_path = args.input.expanduser().resolve()

    if not input_path.exists() or not input_path.is_file():
        raise ValueError(f"Input file does not exist: {input_path}")

    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file extension. Supported extensions: {allowed}")

    if args.threads < 1:
        raise ValueError("--threads must be at least 1.")

    if not 0.0 <= args.overlap < 1.0:
        raise ValueError("--overlap must be in the range [0.0, 1.0).")

    output_dir = (
        args.output.expanduser().resolve()
        if args.output
        else Path.cwd() / f"{input_path.stem}_stems"
    )

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}\n"
            "Choose another --output path or pass --overwrite."
        )

    return input_path, output_dir


def configure_runtime(device: str, threads: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))

    import torch

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot access a CUDA GPU. "
            "Try --device cpu."
        )


def write_float_wav(audio: Any, destination: Path, sample_rate: int) -> None:
    """Write a 32-bit floating-point WAV without per-stem peak normalization."""
    import torch
    import torchaudio

    audio = audio.detach().to(device="cpu", dtype=torch.float32).contiguous()

    if audio.ndim != 2:
        raise RuntimeError(
            f"Unexpected stem tensor shape {tuple(audio.shape)}; expected [channels, frames]."
        )
    if audio.numel() == 0:
        raise RuntimeError(f"Empty stem tensor for {destination.name}.")
    if not torch.isfinite(audio).all().item():
        raise RuntimeError(f"Stem contains NaN or infinity: {destination.name}")

    torchaudio.save(
        str(destination),
        audio,
        sample_rate,
        encoding="PCM_F",
        bits_per_sample=32,
    )


def run_specialist_stem(
    input_path: Path,
    work_dir: Path,
    model_dir: Path,
    model_filename: str,
    target_stem: str,
    target_sr: int,
    target_channels: int,
    target_frames: int,
) -> tuple[Any, float]:
    """Run an audio-separator model and return a [channels, frames] float32 tensor."""
    import gc
    import torch
    import torchaudio
    from audio_separator.separator import Separator

    work_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  [SPECIALIST:{target_stem.upper()}] Loading {model_filename}...")
    started = time.perf_counter()

    # use_soundfile=False avoids passing MP3/compressed input subtypes into WAV writer.
    separator = Separator(
        output_dir=str(work_dir),
        model_file_dir=str(model_dir),
        output_format="WAV",
        output_single_stem=target_stem,
        normalization_threshold=0.9,
        amplification_threshold=0.0,
        use_soundfile=False,
        mdxc_params={
            "segment_size": 256,
            "override_model_segment_size": False,
            "batch_size": 1,
            "overlap": 8,
            "pitch_shift": 0,
        },
        mdx_params={
            "hop_length": 1024,
            "segment_size": 512,
            "overlap": 0.5,
            "batch_size": 1,
            "enable_denoise": False,
        },
        demucs_params={
            "segment_size": "Default",
            "shifts": 2,
            "overlap": 0.5,
            "segments_enabled": True,
        },
    )

    separator.load_model(model_filename=model_filename)
    output_files = separator.separate(str(input_path))
    elapsed = time.perf_counter() - started

    raw_candidates = [Path(p) for p in (output_files or []) if p]
    candidates: list[Path] = []
    for p in raw_candidates:
        resolved = p if p.is_absolute() else (work_dir / p)
        if resolved.is_file():
            candidates.append(resolved)
        elif p.is_file():
            candidates.append(p)

    if not candidates:
        candidates = sorted(work_dir.rglob("*.wav"), key=lambda f: f.stat().st_mtime)

    if not candidates:
        raise RuntimeError(
            f"Specialist {target_stem} model produced no readable output in {work_dir}: {output_files}"
        )

    tensor, sr = torchaudio.load(str(candidates[-1]))
    tensor = tensor.to(dtype=torch.float32)

    if sr != target_sr:
        tensor = torchaudio.functional.resample(tensor, orig_freq=sr, new_freq=target_sr)

    if tensor.shape[0] != target_channels:
        if tensor.shape[0] == 1 and target_channels > 1:
            tensor = tensor.repeat(target_channels, 1)
        elif tensor.shape[0] > target_channels:
            tensor = tensor[:target_channels, :]

    if tensor.shape[-1] > target_frames:
        tensor = tensor[:, :target_frames]
    elif tensor.shape[-1] < target_frames:
        pad_amount = target_frames - tensor.shape[-1]
        tensor = torch.nn.functional.pad(tensor, (0, pad_amount))

    del separator
    gc.collect()
    return tensor.contiguous(), elapsed


def main() -> int:
    args = parse_args()

    if args.list_models:
        return list_specialist_models(args.list_models)

    try:
        input_path, output_dir = validate_args(args)
        configure_runtime(args.device, args.threads)

        from demucs.api import Separator

        output_dir.mkdir(parents=True, exist_ok=True)

        print("\nRAVEGRID STEM SEPARATOR (v2)")
        print(f"Input      : {input_path}")
        print(f"Output     : {output_dir}")
        print(f"Demucs     : {args.model}")
        print(f"Vocal Spec : {args.vocal_model or 'Demucs baseline'}")
        print(f"Bass Spec  : {args.bass_model or 'Demucs baseline'}")
        print(f"Device     : {args.device}")
        print("\nRunning Demucs baseline separation...")

        started = time.perf_counter()
        separator = Separator(
            model=args.model,
            device=args.device,
            shifts=0,
            overlap=args.overlap,
            split=True,
            jobs=0,
            progress=True,
        )

        original, separated = separator.separate_audio_file(str(input_path))
        demucs_seconds = time.perf_counter() - started

        missing = set(STEM_MAPPING) - set(separated.keys())
        if missing:
            raise RuntimeError(
                "The selected Demucs model did not return expected sources: "
                + ", ".join(sorted(missing))
            )

        sample_rate = int(separator.samplerate)
        target_channels = int(original.shape[0])
        target_frames = int(original.shape[-1])

        del separator
        gc.collect()

        specialist_runs: dict[str, dict[str, Any]] = {}
        with tempfile.TemporaryDirectory(prefix="ravegrid_spec_") as tmp_dir:
            tmp_root = Path(tmp_dir)

            if args.vocal_model:
                spec_vocal, spec_sec = run_specialist_stem(
                    input_path=input_path,
                    work_dir=tmp_root / "vocals",
                    model_dir=args.model_dir.expanduser().resolve(),
                    model_filename=args.vocal_model,
                    target_stem="Vocals",
                    target_sr=sample_rate,
                    target_channels=target_channels,
                    target_frames=target_frames,
                )
                separated["vocals"] = spec_vocal
                specialist_runs["vocal"] = {
                    "model": args.vocal_model,
                    "seconds": round(spec_sec, 3),
                }

            if args.bass_model:
                spec_bass, spec_sec = run_specialist_stem(
                    input_path=input_path,
                    work_dir=tmp_root / "bass",
                    model_dir=args.model_dir.expanduser().resolve(),
                    model_filename=args.bass_model,
                    target_stem="Bass",
                    target_sr=sample_rate,
                    target_channels=target_channels,
                    target_frames=target_frames,
                )
                separated["bass"] = spec_bass
                specialist_runs["bass"] = {
                    "model": args.bass_model,
                    "seconds": round(spec_sec, 3),
                }

        stem_report: dict[str, dict[str, Any]] = {}
        print("\nWriting 32-bit float WAV stems...")
        for demucs_name, output_name in STEM_MAPPING.items():
            destination = output_dir / f"{output_name}.wav"
            tensor = separated[demucs_name]
            write_float_wav(tensor, destination, sample_rate)

            source_type = (
                f"specialist:{specialist_runs[output_name]['model']}"
                if output_name in specialist_runs
                else f"demucs:{args.model}"
            )
            stem_report[output_name] = {
                "file": destination.name,
                "source": source_type,
                "channels": int(tensor.shape[0]),
                "frames": int(tensor.shape[-1]),
                "sample_rate": sample_rate,
                "duration_seconds": round(tensor.shape[-1] / sample_rate, 6),
            }
            print(f"  [OK] {destination.name} ({source_type})")

        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_file": str(input_path),
            "demucs_model": args.model,
            "specialist_upgrades": specialist_runs,
            "device": args.device,
            "cpu_threads": args.threads,
            "sample_rate": sample_rate,
            "demucs_seconds": round(demucs_seconds, 3),
            "input_duration_seconds": round(target_frames / sample_rate, 6),
            "stems": stem_report,
            "notes": [
                "'synth' is Demucs' residual 'other' source; it is not guaranteed to contain only synthesizers.",
                "Specialist stems are extracted from the original mix and aligned to Demucs' sample rate, channels, and length.",
                "Stems are saved as 32-bit floating-point WAV without independent peak normalization.",
            ],
        }

        report_path = output_dir / "separation_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        elapsed = time.perf_counter() - started
        print("\nSeparation complete.")
        print(f"Total time   : {elapsed:.1f} seconds")
        print(f"Report       : {report_path}")
        print(f"Output folder: {output_dir}")
        return 0

    except KeyboardInterrupt:
        print("\nCancelled by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
