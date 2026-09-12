"""OpenAI-compatible speech-to-text server backed by faster-whisper.

Serves ``POST /v1/audio/transcriptions`` (multipart, with a JSON fallback)
so Open WebUI (``audio.stt.openai``) and Pipecat's ``OpenAISTTService`` can
point at it unchanged. One model is loaded at startup; the OpenAI
``model`` field is accepted but ignored (a single model is served).

Environment (set by the NixOS module):
    STT_HOST         bind address            (default 127.0.0.1)
    STT_PORT         port                    (default 8100)
    STT_MODEL        faster-whisper model    (default large-v3-turbo)
    STT_DEVICE       auto|cpu|cuda           (default auto)
    STT_COMPUTE_TYPE compute type            (default: float16 on cuda, int8 on cpu)
    STT_VAD_FILTER   enable VAD filter       (default true)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Annotated, Any, Optional

import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from faster_whisper import WhisperModel

log = logging.getLogger("speech_server.stt")

HOST = os.environ.get("STT_HOST", "127.0.0.1")
PORT = int(os.environ.get("STT_PORT", "8100"))
MODEL_NAME = os.environ.get("STT_MODEL", "large-v3-turbo")
DEVICE = os.environ.get("STT_DEVICE", "auto")
COMPUTE_TYPE = os.environ.get("STT_COMPUTE_TYPE", "")
VAD_FILTER = os.environ.get("STT_VAD_FILTER", "true").lower() in ("1", "true", "yes")

app = FastAPI(title="speech-server STT")


def _resolve_backend() -> tuple[str, str]:
    """Pick (device, compute_type); on this host the CTranslate2 aarch64
    wheel is CPU-only, so auto resolves to int8 on the CPU unless CUDA
    is genuinely available."""
    device = DEVICE
    if device == "auto":
        try:
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    compute_type = COMPUTE_TYPE or ("float16" if device == "cuda" else "int8")
    return device, compute_type


@asynccontextmanager
async def lifespan(app: FastAPI):
    device, compute_type = _resolve_backend()
    log.info("loading %s on %s (compute_type=%s)", MODEL_NAME, device, compute_type)
    app.state.model = WhisperModel(MODEL_NAME, device=device, compute_type=compute_type)
    app.state.backend = f"{device}/{compute_type}"
    log.info("model loaded; STT ready on %s:%d", HOST, PORT)
    yield


app = FastAPI(title="speech-server STT", lifespan=lifespan)


def _normalise_language(language: str) -> Optional[str]:
    """Map OpenAI-style language hints to faster-whisper (None = autodetect)."""
    language = (language or "").strip().lower()
    if language in ("", "auto", "none", "und", "en-us", "en_us"):
        # faster-whisper takes ISO-639-1 ("en"); "en-us" is not valid.
        return "en" if language.startswith("en") else None
    return language


def _transcribe(model: WhisperModel, audio: bytes, language: str):
    """Run one transcription; accepts bytes (any format PyAV can decode)."""
    segments, info = model.transcribe(
        io.BytesIO(audio),
        language=None if language in (None, "") else language,
        vad_filter=VAD_FILTER,
        beam_size=5,
    )
    segs = list(segments)
    text = " ".join(s.text.strip() for s in segs)
    return segs, info, text


def _format_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _render_segments(segs, fmt: str) -> str:
    if fmt in ("srt", "vtt"):
        header = "WEBVTT\n\n" if fmt == "vtt" else ""
        lines = [header]
        for i, s in enumerate(segs, start=1):
            lines.append(f"{i}\n{_format_timestamp(s.start)} --> {_format_timestamp(s.end)}\n{s.text.strip()}\n")
        return "\n".join(lines)
    raise ValueError(f"unsupported subtitle format: {fmt}")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "backend": getattr(app.state, "backend", "loading"),
    }


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "owned_by": "speech-server",
            }
        ],
    }


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    request: Request,
    file: Annotated[Optional[UploadFile], File()] = None,
    model: str = Form("speech-stt"),
    language: str = Form(""),
    response_format: str = Form("json"),
    prompt: Optional[str] = Form(None),
    temperature: Optional[float] = Form(None),  # accepted, ignored
    include: Optional[str] = Form(None),  # accepted, ignored
):
    """OpenAI Whisper-shaped transcription endpoint.

    Multipart form (the OpenAI default) is primary; a JSON body carrying
    base64 audio in ``file``/``input_audio`` is also accepted (Open WebUI's
    ``api_request_format`` option).
    """
    if file is not None:
        audio = await file.read()
    else:
        ctype = request.headers.get("content-type", "")
        if not ctype.startswith("application/json"):
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "multipart form file or JSON body required"}},
            )
        body = json.loads(await request.body())
        raw = body.get("file") or body.get("input_audio") or ""
        # Accept raw base64 or a data: URL.
        if "," in raw and raw.startswith("data:"):
            raw = raw.split(",", 1)[1]
        if not raw:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "no audio found (expected file/input_audio)"}},
            )
        audio = base64.b64decode(re.sub(r"\s+", "", raw))

    if not audio:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "empty audio"}},
        )

    whisper_model = app.state.model
    try:
        segs, info, text = _transcribe(whisper_model, audio, _normalise_language(language))
    except Exception as e:  # noqa: BLE001 - report, don't crash the server
        log.exception("transcription failed")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": f"transcription failed: {e}"}},
        )

    if response_format == "text":
        return PlainTextResponse(text)
    if response_format in ("srt", "vtt"):
        return Response(content=_render_segments(segs, response_format), media_type="text/plain")

    payload: dict[str, Any] = {
        "text": text,
        "language": info.language,
        "duration": round(float(info.duration), 3),
    }
    if response_format == "verbose_json":
        payload["segments"] = [
            {
                "id": i,
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "text": s.text.strip(),
                "tokens": list(s.tokens or []),
                "token_probs": [round(t, 6) for t in (s.token_probs or [])],
            }
            for i, s in enumerate(segs)
        ]
    return JSONResponse(payload)


def main() -> None:
    logging.basicConfig(level=os.environ.get("STT_LOG_LEVEL", "INFO"))
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
