from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def band_energy_ratio(y: np.ndarray, sr: int, f_low: float, f_high: float) -> float:
    """Fraction of total signal energy inside [f_low, f_high] Hz."""
    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    total = float(np.sum(stft)) + 1e-12
    mask = (freqs >= f_low) & (freqs <= f_high)
    return float(np.sum(stft[mask])) / total


def classify_hit(y: np.ndarray, sr: int) -> str:
    """Classify a single drum hit into 'kick', 'snare', or 'hat'."""
    low_ratio = band_energy_ratio(y, sr, 20.0, 160.0)
    high_ratio = band_energy_ratio(y, sr, 2500.0, min(14000.0, sr / 2.0))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    # 1. Kick: heavy low-end and not dominated by high frequencies
    if low_ratio >= 0.45 and low_ratio > high_ratio * 1.5:
        return "kick"

    # 2. Hat: bright, high-frequency-heavy transient
    if high_ratio >= 0.35 and centroid >= 2600.0:
        return "hat"

    # 3. Snare: everything else in a standard 3-piece drum groove
    return "snare"


def slice_to_three_kits(
    input_path: Path,
    output_dir: Path,
    min_gap_ms: float = 70.0,
    max_slice_ms: float = 800.0,
    delta: float = 0.15,
) -> dict[str, int]:
    y, sr = librosa.load(str(input_path), sr=None, mono=True)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    wait = max(1, int(librosa.time_to_frames(min_gap_ms / 1000.0, sr=sr, hop_length=512)))

    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=512,
        backtrack=True,
        delta=delta,
        wait=wait,
    )
    times = librosa.frames_to_time(onsets, sr=sr, hop_length=512)

    counts = {"kick": 0, "snare": 0, "hat": 0}
    for inst in counts:
        (output_dir / inst).mkdir(parents=True, exist_ok=True)

    pre = int(sr * 0.005)
    max_len = int(sr * max_slice_ms / 1000.0)

    for i, t in enumerate(times):
        start = max(0, int(t * sr) - pre)
        end = int(times[i + 1] * sr) if i + 1 < len(times) else len(y)
        end = min(end, start + max_len, len(y))
        hit = y[start:end]

        if len(hit) < int(sr * 0.015) or float(np.max(np.abs(hit))) < 0.02:
            continue

        # Short fade-out & peak normalize to -1 dBFS
        fade_len = min(int(sr * 0.01), len(hit) // 2)
        if fade_len > 0:
            hit = hit.copy()
            hit[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)
        hit = hit / float(np.max(np.abs(hit))) * (10.0 ** (-1.0 / 20.0))

        label = classify_hit(hit, sr)
        counts[label] += 1
        out_path = output_dir / label / f"{label}_{counts[label]:02d}.wav"
        sf.write(str(out_path), hit, sr, subtype="PCM_24")

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Slice a drum stem into kick, snare, and hat folders.")
    parser.add_argument("input_file", type=Path)
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("kit_3part"))
    parser.add_argument("--delta", type=float, default=0.15)
    args = parser.parse_args()

    counts = slice_to_three_kits(args.input_file, args.output_dir, delta=args.delta)
    print(f"Saved -> kicks: {counts['kick']}, snares: {counts['snare']}, hats: {counts['hat']}")


if __name__ == "__main__":
    main()
