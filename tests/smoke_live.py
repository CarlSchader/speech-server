"""Opt-in CUDA/model/API smoke test. Uses synthetic text and loopback ports.

Requires SPEECH_ENV, CUDA library path, and model cache from benchmarks. It
starts only its own subprocesses and always stops them, including on failure.
"""

import asyncio
import json
import os
import subprocess
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

import httpx


async def check_websocket(session):
    import websockets
    from pipecat.serializers.protobuf import frame_protos

    uri = f"ws://127.0.0.1:18766/ws-client?token={session['token']}"
    async with websockets.connect(uri, origin="https://voice.carlschader.com") as ws:
        frame = frame_protos.Frame()
        frame.message.data = json.dumps(
            {
                "label": "rtvi-ai",
                "type": "client-ready",
                "id": "smoke",
                "data": {"version": "1.0.0"},
            }
        )
        await ws.send(frame.SerializeToString())
        async with asyncio.timeout(45):
            while True:
                packet = await ws.recv()
                output = frame_protos.Frame.FromString(packet)
                if output.WhichOneof("frame") == "audio" and output.audio.audio:
                    break


def main():
    package = Path(os.environ["SPEECH_ENV"])
    cache = Path(os.environ["SPEECH_BENCH_CACHE"])
    common = os.environ | {"HF_HOME": str(cache / "huggingface")}
    configs = {
        "stt": common
        | {
            "STT_PORT": "18100",
            "STT_DEVICE": "cuda",
            "STT_COMPUTE_TYPE": "int8_float16",
        },
        "tts": common
        | {
            "TTS_PORT": "18880",
            "TTS_DEVICE": "cuda",
            "TTS_MODEL_FILE": str(cache / "kokoro-v1.0.onnx"),
            "TTS_VOICES_FILE": str(cache / "voices-v1.0.bin"),
        },
        "agent": common
        | {
            "AGENT_HOST": "127.0.0.1",
            "AGENT_PORT": "18766",
            "STT_BASE_URL": "http://127.0.0.1:18100/v1",
            "TTS_BASE_URL": "http://127.0.0.1:18880/v1",
            "LLM_BASE_URL": "https://llm.carlschader.com/v1",
            "LLM_EXTRA_BODY": '{"chat_template_kwargs":{"enable_thinking":false}}',
            "PIPECAT_ALLOWED_ORIGINS": "https://voice.carlschader.com",
            "PIPECAT_WEBSOCKET_AUTH": "token",
        },
    }
    processes = []
    with (
        tempfile.TemporaryDirectory(prefix="speech-smoke-") as tmp,
        ExitStack() as stack,
    ):
        try:
            for name, env in configs.items():
                log = stack.enter_context(open(Path(tmp) / f"{name}.log", "w+"))
                args = [str(package / "bin" / f"speech-{name}")]
                if name == "agent":
                    args += [
                        "--transport",
                        "websocket",
                        "--proxy",
                        "voice.carlschader.com",
                    ]
                proc = subprocess.Popen(args, env=env, stdout=log, stderr=log)
                processes.append((proc, log))
            with httpx.Client(timeout=60, follow_redirects=True) as client:
                for port in (18100, 18880, 18766):
                    deadline = time.monotonic() + 90
                    while True:
                        if any(p.poll() is not None for p, _ in processes):
                            raise RuntimeError("service exited")
                        try:
                            r = client.get(
                                f"http://127.0.0.1:{port}/"
                                if port == 18766
                                else f"http://127.0.0.1:{port}/health"
                            )
                            r.raise_for_status()
                            print(port, "ready", flush=True)
                            break
                        except httpx.HTTPError:
                            if time.monotonic() >= deadline:
                                raise
                            time.sleep(0.2)
                start = time.perf_counter()
                r = client.post(
                    "http://127.0.0.1:18100/v1/audio/transcriptions",
                    files={
                        "file": (
                            "sample.wav",
                            (cache / "short.wav").read_bytes(),
                            "audio/wav",
                        )
                    },
                    data={"model": "large-v3-turbo", "response_format": "verbose_json"},
                )
                r.raise_for_status()
                text = r.json()["text"]
                print("STT:", text, time.perf_counter() - start, flush=True)
                start = time.perf_counter()
                r = client.post(
                    "https://llm.carlschader.com/v1/chat/completions",
                    json={
                        "model": "qwen3.8-27b",
                        "messages": [
                            {
                                "role": "system",
                                "content": "Answer in one short sentence. No markdown.",
                            },
                            {"role": "user", "content": text},
                        ],
                        "max_tokens": 80,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                r.raise_for_status()
                answer = r.json()["choices"][0]["message"]["content"]
                print("LLM seconds:", time.perf_counter() - start, flush=True)
                for fmt in ("pcm", "wav", "mp3"):
                    r = client.post(
                        "http://127.0.0.1:18880/v1/audio/speech",
                        json={
                            "model": "tts-1",
                            "input": answer,
                            "voice": "alloy",
                            "response_format": fmt,
                        },
                    )
                    r.raise_for_status()
                    assert len(r.content) > 100
                    print("TTS:", fmt, len(r.content), flush=True)
                r = client.post(
                    "http://127.0.0.1:18766/start",
                    json={"transport": "websocket"},
                    headers={
                        "Origin": "https://voice.carlschader.com",
                        "Host": "voice.carlschader.com",
                    },
                )
                r.raise_for_status()
                session = r.json()
                assert session["wsUrl"] == "wss://voice.carlschader.com/ws-client"
                asyncio.run(check_websocket(session))
                print(
                    "Agent start, WSS URL and synthesized WebSocket audio OK",
                    flush=True,
                )
                denied = client.post(
                    "http://127.0.0.1:18766/start",
                    json={"transport": "websocket"},
                    headers={"Origin": "https://untrusted.example"},
                )
                assert "access-control-allow-origin" not in denied.headers
        finally:
            for proc, _ in processes:
                proc.terminate()
            for proc, log in processes:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                log.seek(0)
                print(log.read())
                log.close()


if __name__ == "__main__":
    main()
