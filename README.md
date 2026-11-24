## HTDemucs ONNX — Four-Stem Separation Test

A small command-line prototype that reproduces the main ideas of [openDAW](https://github.com/andremichelle/openDAW)'s browser-based HTDemucs stem-separation pipeline, using **ONNX Runtime on the CPU**.

Given an audio file, it exports four separate stereo WAV files:

- `drums.wav`
- `bass.wav`
- `other.wav`
- `vocals.wav`

This is an experimental RAVEGRID research/prototyping tool. It is not a complete DAW integration.

## 1. How it works

```text
             Input audio
          (WAV / MP3 / FLAC...)
                  |
                  v
            FFmpeg decode
         stereo, 44,100 Hz
                  |
                  v
          Planar float32 audio
             [2, samples]
                  |
                  v
         HTDemucs v4 ONNX model
          ONNX Runtime (CPU)
                  |
          7.8-second windows
           0.25-second overlap
                  |
                  v
         Crossfade reconstruction
                  |
          +-------+-------+-------+
          |       |       |       |
        Drums    Bass   Other   Vocals
          |       |       |       |
          +-------+-------+-------+
                  |
                  v
             Stereo WAVs
```

### Model and runtime

| Item | Value |
|---|---|
| Separation model | HTDemucs v4 |
| Model format | ONNX |
| Inference runtime | ONNX Runtime |
| Execution provider | `CPUExecutionProvider` |
| Sample rate | 44,100 Hz |
| Input channels | Stereo |
| Input tensor | `[1, 2, 343980]` |
| Window duration | 7.8 seconds |
| Window size | 343,980 samples |
| Overlap | 0.25 seconds |
| Overlap size | 11,025 samples |
| Stem order | drums, bass, other, vocals |

The model is downloaded automatically on first run from the model URL embedded in the script. The script verifies the downloaded model using its SHA-256 checksum and caches it at:

```text
~/.cache/ravegrid/models/htdemucs-v4/model.onnx
```

The model download is approximately 250 MB. Later runs reuse the cached copy.

> **Important:** ONNX Runtime is the inference engine; HTDemucs is the separation model. This prototype currently uses CPU inference. It does not enable openDAW's browser WebGPU execution provider.

## 2. Requirements

- Python 3.10 or newer
- FFmpeg available on your system
- Python packages listed in `requirements.txt`
- Internet access for the initial model download
- Approximately 250 MB of disk space for the model, plus space for input and output audio

On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y ffmpeg python3-venv
```

## 3. Installation

Clone or copy this project, then enter its directory:

```bash
cd path/to/project
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the Python dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Usage

Basic usage:

```bash
python opendaw_onnx_stem_test.py path/to/audio.mp3
```

The output directory is created beside the input file. For example:

```text
audio.mp3
audio_opendaw_onnx_stems/
├── drums.wav
├── bass.wav
├── other.wav
├── vocals.wav
└── separation_report.json
```

Choose a custom output directory:

```bash
python opendaw_onnx_stem_test.py audio.mp3 \
  --output ./results/audio_test
```

Set the number of CPU threads:

```bash
python opendaw_onnx_stem_test.py audio.mp3 --threads 3
```

Use a model file that has already been downloaded and verified:

```bash
python opendaw_onnx_stem_test.py audio.mp3 \
  --model ~/.cache/ravegrid/models/htdemucs-v4/model.onnx
```

See all command-line options:

```bash
python opendaw_onnx_stem_test.py --help
```

## 5. Output files

All four stems are written as stereo, 44.1 kHz, 32-bit floating-point WAV files.

| File | Meaning |
|---|---|
| `drums.wav` | Percussive/drum component |
| `bass.wav` | Bass component |
| `other.wav` | Residual non-drum, non-bass, non-vocal component |
| `vocals.wav` | Vocal component |
| `separation_report.json` | Input/model metadata and timing information |

`other.wav` is **not a synth-only stem**. It is the model's residual `other` category and can contain synths, instruments, effects, and other non-vocal material.

The script does not independently peak-normalize the stems. This helps preserve the model's relative output levels for listening and reconstruction tests.

## 6. Processing details

### Audio decoding

FFmpeg decodes the input and converts it to stereo, 44.1 kHz, float32 PCM. The script converts FFmpeg's interleaved stereo samples into planar channel layout:

```text
Left:  [L0, L1, L2, ...]
Right: [R0, R1, R2, ...]
```

The model input is assembled as:

```text
[batch, channels, samples]
[1,     2,        343980]
```

### Chunking

The model is evaluated on fixed 7.8-second windows. Adjacent windows overlap by 0.25 seconds. This allows longer files to be processed without passing the entire recording to the model at once.

### Reconstruction

Each stem is reconstructed independently for each channel. Overlapping predictions are combined using linear crossfades, then returned to the original input frame count.

### Performance report

`separation_report.json` records:

- Input path and model path
- Model SHA-256
- Sample rate, channel count, duration, and frame count
- Window and overlap settings
- Number of chunks
- Execution provider and CPU thread count
- Inference/reconstruction elapsed time
- Real-time factor (RTF)

RTF is calculated as:

```text
RTF = processing time / audio duration
```

For example, an RTF of `2.0` means processing took about twice as long as the audio's playback duration. Lower is faster.

## 7. Comparing separation quality

For a meaningful comparison with another separator:

1. Use the exact same original input file.
2. Keep the sample rate and channel layout consistent.
3. Compare all four stems at matched playback loudness.
4. Listen for leakage, missing transients, phasey artifacts, and musical detail.
5. Sum the four stems and compare the result with the original mix.
6. Keep the original stems unchanged; treat processed alternatives as separate candidates.

Useful listening checks for electronic music:

- **Drums:** kick/snare/hat transient clarity and bleed.
- **Bass:** note definition, low-frequency stability, and distortion/artifacts.
- **Vocals:** synth or reverb leakage into the vocal stem.
- **Other:** completeness of synths, pads, effects, and melodic elements.

A good-sounding isolated stem does not necessarily guarantee accurate reconstruction. Use listening tests and signal measurements together.

## 8. Current limitations

- CPU inference only; no WebGPU provider in this terminal script.
- Stereo input/output only.
- The script expects the model's fixed input/output tensor dimensions and stem ordering.
- `other` is a residual category, not a dedicated synthesizer separator.
- Separation quality is limited by the model and source mix; post-processing cannot reliably recover information the model did not isolate.
- The first run requires downloading the model.
- This is a test harness, not a real-time plugin or production audio engine.

## 9. Troubleshooting

### `ffmpeg was not found`

Install FFmpeg:

```bash
sudo apt install ffmpeg
```

Verify:

```bash
ffmpeg -version
```

### `No module named onnxruntime`

Activate the project's virtual environment and install dependencies:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Model download or checksum error

Check your internet connection and available disk space. If the cached model is incomplete or has an invalid checksum, remove it and rerun:

```bash
rm ~/.cache/ravegrid/models/htdemucs-v4/model.onnx
python opendaw_onnx_stem_test.py path/to/audio.mp3
```

### Processing is too slow

Try a smaller thread count or a different value with `--threads`. Performance depends on the CPU, memory bandwidth, ONNX Runtime build, and audio duration. Avoid running multiple model instances concurrently on a low-core-count machine.

### Unexpected ONNX tensor shape

The script checks the model's input and output shapes. If a different ONNX export is supplied, its tensor layout or output structure may not match this adapter. Inspect the printed model input/output metadata before changing the stem extraction logic.

## 10. Project direction

This prototype is intended to help evaluate a replaceable stem-separation backend for RAVEGRID.

A future architecture can isolate model-specific inference behind a common interface:

```text
                 Stem Separation API
                          |
                  Model Adapter
                          |
             +------------+------------+
             |            |            |
          HTDemucs     BS-RoFormer   Other model
            ONNX         ONNX         adapter
             |            |            |
             +------------+------------+
                          |
                    Stem outputs
                          |
                 Stem analysis/editing
                          |
                     DAW tracks
```

The DAW should consume a stable four-stem result contract without depending on the implementation details of a particular model or runtime.

## References

- [openDAW repository](https://github.com/andremichelle/openDAW)
- [ONNX Runtime documentation](https://onnxruntime.ai/docs/)
- [HTDemucs ONNX model](https://assets.opendaw.studio/models/htdemucs/v4/model.onnx)

The model URL, expected checksum, model dimensions, and processing parameters are defined in `opendaw_onnx_stem_test.py`. Check the upstream project and model terms before redistributing the model or shipping it in a product.
"""

path = Path("/mnt/data/README.md")
path.write_text(readme, encoding="utf-8")
print(f"Created {path} ({len(readme.splitlines())} lines)")
