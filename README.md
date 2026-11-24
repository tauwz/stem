Here is a `README.md` tailored to your `stem_separator.py` script:

```markdown
# RAVEGRID Stem Separator

A Python CLI stem separator built around **Demucs** (`htdemucs` / `htdemucs_ft`) with optional **specialist model upgrades** via `audio-separator` for cleaner vocals and bass. Outputs are written as **32-bit floating-point WAV** files without per-stem peak normalization, along with a `separation_report.json` summary.

---

## Features

- **Baseline 4-stem separation with Demucs** (`htdemucs` or `htdemucs_ft`).
- **Optional specialist upgrades** for `vocals` (`--vocal-model`) and `bass` (`--bass-model`) run against the original mix and aligned to Demucs' sample rate, channel count, and frame length.
- **Model discovery flag** (`--list-models vocals|bass`) to inspect available specialist models.
- **32-bit float (`PCM_F`) WAV output** with no independent peak normalization so stem balance stays intact.
- **CPU-friendly controls** (`--device`, `--threads`, `--overlap`, `--model-dir`, `--overwrite`).

---

## Requirements

- **Python** with `torch`, `torchaudio`, and `demucs` installed.
- **`audio-separator`** (optional, required only when using `--list-models`, `--vocal-model`, or `--bass-model`).
- **FFmpeg** installed on your system for compressed audio decoding (`.mp3`, `.m4a`, `.aac`, `.ogg`).

---

## Output Structure

By default, stems are written to `./<input-name>_stems/`:

```text
<input-name>_stems/
├── drumkit.wav           # Demucs "drums"
├── bass.wav              # Demucs "bass" (or specialist bass if --bass-model is set)
├── synth.wav             # Demucs residual "other" (not guaranteed synth-only)
├── vocal.wav             # Demucs "vocals" (or specialist vocals if --vocal-model is set)
└── separation_report.json

```

---

## Usage

### Basic 4-stem separation

```bash
python stem_separator.py /path/to/audio.mp3

```

### Use `htdemucs_ft` baseline

```bash
python stem_separator.py /path/to/audio.wav --model htdemucs_ft

```

### List available specialist models

```bash
python stem_separator.py --list-models bass
python stem_separator.py --list-models vocals

```

### Upgrade vocals and/or bass with specialist models

```bash
python stem_separator.py /path/to/audio.mp3 \
  --vocal-model model_bs_roformer_ep_317_sdr_12.9755.ckpt \
  --bass-model kuielab_a_bass.onnx \
  --overwrite

```

---

## CLI Options

| Flag | Default | Description |
| --- | --- | --- |
| `input` | *required* | Path to input audio (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`). |
| `--list-models` | `None` | List candidate `audio-separator` models for `vocals` or `bass` and exit. |
| `--model` | `htdemucs` | Demucs baseline model (`htdemucs` or `htdemucs_ft`). |
| `--vocal-model` | `None` | Optional `audio-separator` vocal model filename to replace `vocal.wav`. |
| `--bass-model` | `None` | Optional `audio-separator` bass model filename to replace `bass.wav`. |
| `--model-dir` | `~/.cache/ravegrid/audio-separator-models` | Cache directory for specialist model weights. |
| `--device` | `cpu` | Inference device (`cpu` or `cuda`). |
| `--threads` | `3` | Maximum PyTorch CPU threads (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`). |
| `--overlap` | `0.25` | Overlap ratio between Demucs segments (`[0.0, 1.0)`). |
| `--output` | `./<input-name>_stems` | Custom output directory. |
| `--overwrite` | `False` | Allow writing into a non-empty output directory. |

---

## Notes

* **`synth.wav` is Demucs' `other` source**, which may contain synths, guitars, effects, or other residual elements.
* **Specialist models run on the original input**, then are resampled, channel-matched, and padded/trimmed to match Demucs' output dimensions.
* **CPU runtime**: Enabling `--vocal-model`, `--bass-model`, `htdemucs_ft`, or higher overlap/segment settings increases CPU time and memory usage.

```

```
