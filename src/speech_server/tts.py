"""OpenAI-compatible text-to-speech server backed by Kokoro-82M (kokoro-onnx).

Serves ``POST /v1/audio/speech`` so Open WebUI (``audio.tts.openai``) and
Pipecat's ``OpenAITTSService`` can point at it unchanged. OpenAI voice ids
(``alloy``, ``echo``, ...) are mapped to Kokoro voices; any name that
matches a Kokoro voice exactly is used as-is; anything else falls back to
the configured default voice.

Environment (set by the NixOS module):
    TTS_HOST         bind address                  (default 127.0.0.1)
    TTS_PORT         port                          (default 8880)
    TTS_VOICE        default voice                 (default af_heart)
    TTS_MODEL_FILE   path to kokoro-v1.0.onnx      (default ./models/kokoro-v1.0.onnx)
    TTS_VOICES_FILE  path to voices-v1.0.bin       (default ./models/voices-v1.0.bin)

Model files are downloaded from the kokoro-onnx GitHub releases on first
start if missing (same pattern as SGLang pulling weights).
"""

from __future__ import annotations

import io
import json
import logging
import os
import urllib.request
from contextlib import asynccontextmanager
from fractions import Fraction
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

log = logging.getLogger("speech_server.tts")

HOST = os.environ.get("TTS_HOST", "127.0.0.1")
PORT = int(os.environ.get("TTS_PORT", "8880"))
DEFAULT_VOICE = os.environ.get("TTS_VOICE", "af_heart")
MODEL_FILE = os.environ.get("TTS_MODEL_FILE", "models/kokoro-v1.0.onnx")
VOICES_FILE = os.environ.get("TTS_VOICES_FILE", "models/voices-v1.0.bin")

# Latest kokoro-onnx model-files release (contains the v1.0 model + voices).
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.0.bin"

# OpenAI voice ids -> Kokoro voices, so OWUI's default "alloy" just works.
OPENAI_VOICE_ALIASES = {
    "alloy": "af_heart",
    "echo": "bm_daniel",
    "fable": "am_fenrir",
    "onyx": "am_adam",
    "nova": "af_nova",
    "shimmer": "am_alice",
    "coral": "bf_emma",
    "sage": "bm_george",
}

# Kokoro voice family -> espeak-ng language code used for phonemization.
VOICE_LANG = {
    "af": "en-us",
    "am": "en-us",
    "bf": "en-gb",
    "bm": "en-gb",
    "is": "en-us",  # Irish voices; en-us phonemization is a safe fallback
    "em": "es-es",
    "ef": "fr-fr",
    "if": "fr-fr",
    "jf": "ja",
    "jm": "ja",
    "zf": "cmn",
}

MAX_INPUT_CHARS = 20_000


def _download_file(url: str, dest: str) -> None:
    log.info("downloading %s -> %s", url, dest)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    os.replace(tmp, dest)
    log.info("downloaded %s", dest)


def _ensure_model_files() -> None:
    for path, url in ((MODEL_FILE, MODEL_URL), (VOICES_FILE, VOICES_URL)):
        if not os.path.exists(path):
            _download_file(url, path)


class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str = Field(..., min_length=1)
    voice: str = "alloy"
    response_format: str = "mp3"
    speed: float = 1.0
    instructions: str | None = None  # accepted, ignored (Kokoro has no steering)


def _resolve_voice(available: set[str], requested: str) -> str:
    v = (requested or "").strip()
    if v in available:
        return v
    aliased = OPENAI_VOICE_ALIASES.get(v.lower())
    if aliased and aliased in available:
        if aliased != v:
            log.info("mapping OpenAI voice %r -> %s", v, aliased)
        return aliased
    if aliased:
        log.warning("aliased voice %s not in model; using default %s", aliased, DEFAULT_VOICE)
    else:
        log.warning("unknown voice %r; using default %s", requested, DEFAULT_VOICE)
    return DEFAULT_VOICE


def _lang_for_voice(voice: str) -> str:
    return VOICE_LANG.get(voice.split("_", 1)[0].lower(), "en-us")


def _to_s16(audio: np.ndarray) -> bytes:
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype("<i2")
    return pcm.tobytes()


def _encode_mp3(pcm: bytes, sample_rate: int) -> bytes:
    import av

    arr = np.frombuffer(pcm, dtype=np.int16).reshape(1, -1)  # (channels, samples)
    frame = av.AudioFrame.from_ndarray(arr, format="s16", layout="mono")
    frame.time_base = Fraction(1, sample_rate)
    # from_ndarray leaves sample_rate at 0; PyAV's encoder resampler builds
    # an abuffer filter from it, so an unset rate makes graph init fail.
    frame.sample_rate = sample_rate
    out = io.BytesIO()
    container = av.open(out, mode="w", format="mp3")
    stream = container.add_stream("libmp3lame", rate=sample_rate)
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode(None):  # flush
        container.mux(packet)
    container.close()
    return out.getvalue()


def _encode_wav(pcm: bytes, sample_rate: int) -> bytes:
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from kokoro_onnx import Kokoro

    _ensure_model_files()
    log.info("loading kokoro from %s", MODEL_FILE)
    app.state.kokoro = Kokoro(MODEL_FILE, VOICES_FILE)
    log.info("kokoro ready; voices: %s", ", ".join(app.state.kokoro.get_voices()))
    yield


app = FastAPI(title="speech-server TTS", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "model": "kokoro-v1.0"}


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "owned_by": "speech-server"}
            for m in ("tts-1", "kokoro", "kokoro-v1.0")
        ],
    }


@app.get("/v1/voices")
async def voices() -> dict[str, Any]:
    return {"voices": app.state.kokoro.get_voices()}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest) -> Response:
    """Blocking synthesis (runs in FastAPI's thread pool); Kokoro-82M on CPU
    is near real-time, so a few concurrent requests are fine."""
    kokoro = app.state.kokoro

    if len(req.input) > MAX_INPUT_CHARS:
        raise HTTPException(400, f"input exceeds {MAX_INPUT_CHARS} characters")
    if req.response_format not in ("mp3", "wav", "pcm"):
        raise HTTPException(
            400, f"response_format must be one of mp3, wav, pcm (got {req.response_format!r})"
        )
    speed = min(max(req.speed, 0.5), 2.0)  # kokoro-onnx valid range

    voice = _resolve_voice(set(kokoro.get_voices()), req.voice)
    audio, sample_rate = kokoro.create(
        req.input,
        voice=voice,
        speed=speed,
        lang=_lang_for_voice(voice),
    )
    pcm = _to_s16(audio)

    if req.response_format == "mp3":
        body, media = _encode_mp3(pcm, sample_rate), "audio/mpeg"
    elif req.response_format == "wav":
        body, media = _encode_wav(pcm, sample_rate), "audio/wav"
    else:
        # Raw 16-bit LE mono PCM at sample_rate — what pipecat's
        # OpenAITTSService (response_format="pcm") expects.
        body, media = pcm, "audio/pcm"
    return Response(content=body, media_type=media)


def main() -> None:
    logging.basicConfig(level=os.environ.get("TTS_LOG_LEVEL", "INFO"))
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
