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
