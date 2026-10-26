"""Measure paired speech latency while the existing image API generates images.

Uses synthetic fixtures from benchmark.py. Runs image requests sequentially;
never changes or restarts the production image service. Baseline uses no speech.
"""

import argparse
import concurrent.futures
import json
import os
import resource
import statistics
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from benchmark import TEXTS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--mode", choices=["baseline", "cpu", "mixed", "cuda"], required=True
    )
    p.add_argument("--images", type=int, default=3)
    p.add_argument(
        "--image-url", default="https://img.carlschader.com/v1/images/generations"
    )
    a = p.parse_args()
    os.environ["HF_HOME"] = str(a.cache / "huggingface")
    records = []
    stop = threading.Event()
    gpu = []
    available_ram = []

    def monitor():
        while not stop.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=3,
                )
                gpu.append(int(out.strip()))
                for line in Path("/proc/meminfo").read_text().splitlines():
                    if line.startswith("MemAvailable:"):
                        available_ram.append(int(line.split()[1]) / 1024)
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            stop.wait(0.25)

    if a.mode != "baseline":
        import onnxruntime as ort
        from faster_whisper import WhisperModel
        from kokoro_onnx import Kokoro

        stt = WhisperModel(
            "large-v3-turbo",
            device="cpu" if a.mode == "cpu" else "cuda",
            compute_type="int8" if a.mode == "cpu" else "int8_float16",
            cpu_threads=12 if a.mode == "cpu" else 4,
        )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4 if a.mode == "cuda" else 12
        opts.inter_op_num_threads = 1
        provider = (
            "CUDAExecutionProvider" if a.mode == "cuda" else "CPUExecutionProvider"
        )
        session = ort.InferenceSession(
            str(a.cache / "kokoro-v1.0.onnx"), sess_options=opts, providers=[provider]
        )
        assert session.get_providers()[0] == provider
        tts = Kokoro.from_session(session, str(a.cache / "voices-v1.0.bin"))

        def speech(name):
            start = time.perf_counter()
            segments, _ = stt.transcribe(
                str(a.cache / f"{name}.wav"),
                language="en",
                beam_size=5,
                vad_filter=True,
            )
            text = " ".join(s.text.strip() for s in segments)
            stt_time = time.perf_counter() - start
            start = time.perf_counter()
            audio, rate = tts.create(
                TEXTS[name], voice="af_heart", speed=1.0, lang="en-us"
            )
            return {
                "case": name,
                "stt_seconds": stt_time,
                "tts_seconds": time.perf_counter() - start,
                "text": text,
                "audio_seconds": len(audio) / rate,
            }

        speech("short")
        speech("long")

    def image():
        data = json.dumps(
            {
                "model": "qwen-image-2.1-uc",
                "prompt": "A blue ceramic teapot on a wooden table, studio photograph",
                "n": 1,
                "size": "1024x1024",
            }
        ).encode()
        req = urllib.request.Request(
            a.image_url, data=data, headers={"Content-Type": "application/json"}
        )
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=300) as response:
            body = json.load(response)
        assert body["data"][0].get("b64_json") or body["data"][0].get("url"), (
            "no image returned"
        )
        return time.perf_counter() - start

    sampler = threading.Thread(target=monitor, daemon=True)
    sampler.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            for i in range(a.images):
                work = pool.submit(image)
                samples = []
                if a.mode != "baseline":
                    while not work.done():
                        sample = speech("short" if len(samples) % 2 == 0 else "long")
                        sample["image_still_running_at_end"] = not work.done()
                        samples.append(sample)
                image_seconds = work.result()
                records.append({"image_seconds": image_seconds, "speech": samples})
                print(
                    f"{a.mode} image {i + 1}: {image_seconds:.2f}s, {len(samples)} speech pairs",
                    flush=True,
                )
    finally:
        stop.set()
        sampler.join(timeout=5)
    result = {
        "mode": a.mode,
        "records": records,
        "median_image_seconds": statistics.median(r["image_seconds"] for r in records),
        "sampled_peak_total_gpu_mib": max(gpu, default=0),
        "min_available_ram_mib": min(available_ram, default=0),
        "max_speech_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
    }
    a.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
