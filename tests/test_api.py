"""Model-free API regression tests; run with the packaged Python environment."""

import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import numpy as np
from speech_server import stt, tts


class SpeechTests(unittest.IsolatedAsyncioTestCase):
    async def test_transcription_runs_off_loop_and_verbose_json(self):
        entered = threading.Event()
        release = threading.Event()
        stt.app.state.model = object()
        stt.app.state.backend = "test"
        stt.app.state.inference_lock = asyncio.Lock()
        segment = SimpleNamespace(
            start=0, end=1, text="Hello", tokens=[1], avg_logprob=-0.1
        )

        def transcribe(*args):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("health could not run during inference")
            return [segment], SimpleNamespace(language="en", duration=1), "Hello"

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=stt.app), base_url="http://test"
        ) as client:
            with patch.object(stt, "_transcribe", transcribe):
                task = asyncio.create_task(
                    client.post(
                        "/v1/audio/transcriptions",
                        files={"file": ("sample.wav", b"synthetic")},
                        data={"response_format": "verbose_json"},
                    )
                )
                try:
                    await asyncio.wait_for(asyncio.to_thread(entered.wait), timeout=2)
                    response = await asyncio.wait_for(client.get("/health"), timeout=1)
                    self.assertEqual(response.status_code, 200)
                finally:
                    release.set()
                response = await task
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["segments"][0]["avg_logprob"], -0.1)

    async def test_tts_formats(self):
        tts.app.state.backend = "test"
        tts.app.state.inference_lock = threading.Lock()
        tts.app.state.kokoro = SimpleNamespace(
            get_voices=lambda: ["af_heart"],
            create=lambda *a, **kw: (np.zeros(2400, dtype=np.float32), 24000),
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=tts.app), base_url="http://test"
        ) as client:
            for fmt, content_type in [
                ("wav", "audio/wav"),
                ("mp3", "audio/mpeg"),
                ("pcm", "audio/pcm"),
            ]:
                response = await client.post(
                    "/v1/audio/speech",
                    json={"input": "Hello", "voice": "alloy", "response_format": fmt},
                )
                self.assertEqual(response.status_code, 200)
                self.assertIn(content_type, response.headers["content-type"])
                self.assertGreater(len(response.content), 0)

    async def test_cuda_fallback_fails_startup(self):
        import onnxruntime

        with (
            patch.object(tts, "DEVICE", "cuda"),
            patch.object(tts, "_ensure_model_files"),
            patch.object(onnxruntime, "InferenceSession") as ctor,
        ):
            ctor.return_value.get_providers.return_value = ["CPUExecutionProvider"]
            with self.assertRaisesRegex(RuntimeError, "fell back"):
                async with tts.lifespan(tts.app):
                    self.fail("must not accept CPU fallback")


if __name__ == "__main__":
    unittest.main()
