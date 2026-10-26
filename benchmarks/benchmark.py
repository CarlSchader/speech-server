"""Synthetic, sequential CPU/CUDA speech benchmark (no production writes).

Run in the packaged speech environment. --prepare downloads models and creates
synthetic WAV fixtures. Each measured invocation is a fresh process; model load,
first inference and subsequent warm inference are reported separately. GPU memory
sampling is process-specific; results are not a substitute for listening tests.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import subprocess
import threading
import time
import wave
from pathlib import Path

TEXTS = {
    "short": "Please tell me the weather forecast for tomorrow morning.",
    "long": (
        "Welcome to this synthetic speech benchmark. We are comparing speech recognition "
        "and voice synthesis on a desktop computer. The language model runs on a separate "
        "machine, while image generation shares this graphics card. Please keep the "
        "conversation clear and concise. We will measure response time and memory use "
        "without recording any private conversations."
    ),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task", choices=["prepare", "stt", "tts"], required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--compute", default="int8")
    p.add_argument("--runs", type=int, default=5)
    args = p.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(args.cache / "huggingface")
    os.environ["TTS_MODEL_FILE"] = str(args.cache / "kokoro-v1.0.onnx")
    os.environ["TTS_VOICES_FILE"] = str(args.cache / "voices-v1.0.bin")
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    import numpy as np
    from speech_server import tts

    result = {
        "task": args.task,
        "device": args.device,
        "threads": args.threads,
        "compute": args.compute,
        "runs": args.runs,
        "cases": {},
    }
    stop = threading.Event()
    samples = []

    def sample_gpu():
        while not stop.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=3,
                )
                for line in out.splitlines():
                    pid, mem = line.split(",")
                    if int(pid.strip()) == os.getpid():
                        samples.append(int(mem.strip()))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            stop.wait(0.25)

    monitor = threading.Thread(target=sample_gpu, daemon=True)
    monitor.start()
    try:
        started = time.perf_counter()
        if args.task in ("tts", "prepare"):
            import onnxruntime as ort
            from kokoro_onnx import Kokoro

            tts._ensure_model_files()
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = args.threads
            opts.inter_op_num_threads = 1
            provider = (
                "CUDAExecutionProvider"
                if args.device == "cuda"
                else "CPUExecutionProvider"
            )
            session = ort.InferenceSession(
                tts.MODEL_FILE, sess_options=opts, providers=[provider]
            )
            assert session.get_providers()[0] == provider, (
                "GPU provider silently fell back to CPU"
            )
            model = Kokoro.from_session(session, tts.VOICES_FILE)
            result["providers"] = session.get_providers()
        else:
            from faster_whisper import WhisperModel

            model = WhisperModel(
                "large-v3-turbo",
                device=args.device,
                compute_type=args.compute,
                cpu_threads=args.threads,
                num_workers=1,
            )
        result["model_load_seconds"] = time.perf_counter() - started
        for name, text in TEXTS.items():
            wav = args.cache / f"{name}.wav"
            timings = []
            outputs = []
            for run in range(1 if args.task == "prepare" else args.runs + 1):
                started = time.perf_counter()
                if args.task in ("tts", "prepare"):
                    audio, rate = model.create(
                        text, voice="af_heart", speed=1.0, lang="en-us"
                    )
                    assert np.isfinite(audio).all() and len(audio) > 0
                    encoded = tts._encode_wav(tts._to_s16(audio), rate)
                    duration = len(audio) / rate
                    output = {"audio_seconds": duration, "samples": len(audio)}
                    if args.task == "prepare":
                        wav.write_bytes(encoded)
                else:
                    segments, _info = model.transcribe(
                        str(wav), language="en", vad_filter=True, beam_size=5
                    )
                    transcript = " ".join(s.text.strip() for s in segments)
                    with wave.open(str(wav)) as w:
                        duration = w.getnframes() / w.getframerate()
                    output = {"text": transcript, "audio_seconds": duration}
                timings.append(time.perf_counter() - started)
                outputs.append(output)
            warm = timings[1:] or timings
            result["cases"][name] = {
                "first_seconds": timings[0],
                "warm_seconds": warm,
                "median_seconds": statistics.median(warm),
                "max_seconds": max(warm),
                "median_rtf": statistics.median(warm) / duration,
                "outputs": outputs,
            }
    finally:
        stop.set()
        monitor.join(timeout=5)
    result["max_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    result["sampled_peak_gpu_mib"] = max(samples, default=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
