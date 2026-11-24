#!/usr/bin/env python3
"""Terminal HTDemucs v4 ONNX stem separator following openDAW's pipeline."""
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess, sys, time, urllib.request
from pathlib import Path
import numpy as np
import onnxruntime as ort
import soundfile as sf

SR = 44100
CHANNELS = 2
WINDOW = round(7.8 * SR)       # 343,980 samples
OVERLAP = round(0.25 * SR)     # 11,025 samples
STEMS = ("drums", "bass", "other", "vocals")
MODEL_URL = "https://assets.opendaw.studio/models/htdemucs/v4/model.onnx"
MODEL_SHA256 = "d2b401f322558cd57d67a752ed7be3fa55178a0626011eda8ac7bb74e17280c0"
DEFAULT_MODEL = Path.home() / ".cache/ravegrid/models/htdemucs-v4/model.onnx"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def get_model(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        actual = sha256(path)
        if actual != MODEL_SHA256:
            raise RuntimeError(f"Model SHA-256 mismatch: expected {MODEL_SHA256}, got {actual}")
        print(f"Using cached model: {path}")
        return
    temp = path.with_suffix(path.suffix + ".download")
    print(f"Downloading openDAW HTDemucs model to {path} (about 250 MB)...")
    try:
        req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "RAVEGRID-ONNX-Test/1.0"})
        with urllib.request.urlopen(req, timeout=60) as response, temp.open("wb") as out:
            total = int(response.headers.get("Content-Length", 0))
            done = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
                done += len(block)
                if total:
                    print(f"\r  {done/1e6:.1f}/{total/1e6:.1f} MB ({100*done/total:.1f}%)", end="", flush=True)
        print()
        actual = sha256(temp)
        if actual != MODEL_SHA256:
            raise RuntimeError(f"Downloaded model checksum mismatch: {actual}")
        temp.replace(path)
        print("Download verified.")
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def decode_audio(path: Path) -> np.ndarray:
    """Decode any ffmpeg-supported input to planar float32 stereo at 44.1 kHz."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required. Install it: sudo apt install ffmpeg")
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
           "-acodec", "pcm_f32le", "-ar", str(SR), "-ac", str(CHANNELS), "pipe:1"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode:
        raise RuntimeError("ffmpeg decode failed:\n" + p.stderr.decode(errors="replace"))
    interleaved = np.frombuffer(p.stdout, dtype="<f4")
    if not interleaved.size or interleaved.size % CHANNELS:
        raise RuntimeError("Decoded audio is empty or has an invalid stereo frame count.")
    planar = interleaved.reshape(-1, CHANNELS).T.copy()
    if not np.isfinite(planar).all():
        raise RuntimeError("Input contains NaN or infinite samples.")
    return planar


def plan_chunks(length: int) -> list[int]:
    stride = WINDOW - OVERLAP
    starts, pos = [], 0
    while pos + WINDOW <= length:
        starts.append(pos)
        pos += stride
    if not starts or starts[-1] + WINDOW < length:
        starts.append(max(0, length - WINDOW))
    return starts


def crossfade_chunks(windows: list[np.ndarray], starts: list[int], total: int) -> np.ndarray:
    """Recombine one channel's chunk predictions using openDAW-style linear fades."""
    out = np.zeros(total, dtype=np.float32)
    if not windows:
        return out
    last = len(windows) - 1
    for k, win in enumerate(windows):
        start, n = starts[k], len(win)
        left = 0 if k == 0 else max(0, min(n, starts[k-1] + len(windows[k-1]) - start))
        right = 0 if k == last else max(0, min(n, start + n - starts[k+1]))
        for i in range(n):
            target = start + i
            if not 0 <= target < total:
                continue
            if left and i < left:
                weight = i / left
                out[target] += win[i] * weight
            elif right and i >= n - right:
                offset = i - (n - right)
                weight = 1.0 - offset / right
                out[target] += win[i] * weight
            else:
                out[target] = win[i]
    return out


def make_session(model_path: Path, threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    print(f"ONNX Runtime {ort.__version__}; providers: {ort.get_available_providers()}")
    print(f"Using CPUExecutionProvider with {threads} intra-op threads.")
    return ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])


def infer_chunk(session: ort.InferenceSession, chunk: np.ndarray) -> np.ndarray:
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1:
        raise RuntimeError(f"Expected one model input, got {[x.name for x in inputs]}")
    result = session.run(None, {inputs[0].name: np.ascontiguousarray(chunk[None], dtype=np.float32)})
    if len(result) == 1:
        data = np.asarray(result[0], dtype=np.float32)
        expected = (1, 4, CHANNELS, WINDOW)
        if data.shape != expected:
            raise RuntimeError(f"Unexpected single output shape {data.shape}; expected {expected}. Outputs: {[(x.name,x.shape) for x in outputs]}")
        return data[0]
    if len(result) >= 4:
        rows = []
        for meta, value in zip(outputs[:4], result[:4]):
            arr = np.asarray(value, dtype=np.float32)
            if arr.shape == (1, CHANNELS, WINDOW):
                arr = arr[0]
            if arr.shape != (CHANNELS, WINDOW):
                raise RuntimeError(f"Unexpected output {meta.name} shape {arr.shape}")
            rows.append(arr)
        return np.stack(rows)
    raise RuntimeError(f"Unsupported ONNX output count: {len(result)}")


def separate(audio: np.ndarray, session: ort.InferenceSession) -> tuple[dict[str, np.ndarray], int]:
    total = audio.shape[1]
    starts = plan_chunks(total)
    # stem -> channel -> list of chunk arrays
    collected = {name: [[], []] for name in STEMS}
    for index, start in enumerate(starts):
        chunk = np.zeros((CHANNELS, WINDOW), dtype=np.float32)
        count = max(0, min(WINDOW, total - start))
        if count:
            chunk[:, :count] = audio[:, start:start+count]
        prediction = infer_chunk(session, chunk)  # [stem, channel, sample]
        for s, name in enumerate(STEMS):
            for ch in range(CHANNELS):
                collected[name][ch].append(prediction[s, ch].copy())
        print(f"\rInference: {index+1}/{len(starts)} chunks", end="", flush=True)
    print()
    result = {}
    for name in STEMS:
        result[name] = np.stack([
            crossfade_chunks(collected[name][ch], starts, total)
            for ch in range(CHANNELS)
        ])
    return result, len(starts)


def main() -> int:
    parser = argparse.ArgumentParser(description="openDAW-style HTDemucs v4 ONNX four-stem test")
    parser.add_argument("input", type=Path, help="Input audio: WAV, MP3, FLAC, etc.")
    parser.add_argument("-o", "--output", type=Path, help="Output folder (default: <input>_opendaw_onnx_stems)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help=f"Existing ONNX model path (default: {DEFAULT_MODEL})")
    parser.add_argument("--threads", type=int, default=3, help="CPU threads (default: 3)")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Input file not found: {args.input}")
    if args.threads < 1:
        parser.error("--threads must be >= 1")
    outdir = args.output or args.input.with_name(args.input.stem + "_opendaw_onnx_stems")
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        get_model(args.model)
        audio = decode_audio(args.input)
        duration = audio.shape[1] / SR
        print(f"Input: {args.input} | {duration:.2f}s | {SR} Hz | stereo | {audio.shape[1]:,} frames")
        print(f"Window: 7.8s ({WINDOW:,} samples); overlap: 0.25s ({OVERLAP:,} samples)")
        print("Loading ONNX model...")
        session = make_session(args.model, args.threads)
        print("Inputs:", [(x.name, x.shape, x.type) for x in session.get_inputs()])
        print("Outputs:", [(x.name, x.shape, x.type) for x in session.get_outputs()])
        started = time.perf_counter()
        stems, chunks = separate(audio, session)
        elapsed = time.perf_counter() - started
        for name, planar in stems.items():
            path = outdir / f"{name}.wav"
            sf.write(path, np.ascontiguousarray(planar.T), SR, subtype="FLOAT")
            print(f"Saved {path}")
        report = {
            "input": str(args.input.resolve()), "model": str(args.model.resolve()),
            "model_sha256": sha256(args.model), "sample_rate": SR, "channels": CHANNELS,
            "duration_seconds": duration, "frames": int(audio.shape[1]),
            "segment_seconds": 7.8, "segment_samples": WINDOW,
            "overlap_seconds": 0.25, "overlap_samples": OVERLAP, "chunks": chunks,
            "stems": list(STEMS), "provider": "CPUExecutionProvider", "threads": args.threads,
            "inference_and_stitch_seconds": elapsed,
            "real_time_factor": elapsed / duration if duration else None,
        }
        (outdir / "separation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nFinished in {elapsed:.2f}s (RTF {elapsed/duration:.2f}). Output: {outdir.resolve()}")
        return 0
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
