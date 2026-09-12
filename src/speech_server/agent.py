"""Pipecat voice agent: speech -> STT -> LLM -> TTS -> speech.

Runs pipecat's dev runner (FastAPI + prebuilt browser client) with a pipeline
that chains:

    mic audio -> Silero VAD -> OpenAISTTService (local speech-stt)
              -> OpenAILLMService (SGLang) -> OpenAITTSService (local speech-tts)
              -> speaker audio

The STT/LLM/TTS backends are all OpenAI-compatible and fully configurable via
environment variables, so the agent is engine-agnostic.

Environment (set by the NixOS module):
    AGENT_HOST             bind address                     (default 0.0.0.0)
    AGENT_PORT             runner port                      (default 8765)
    LLM_BASE_URL           OpenAI-compatible LLM base URL   (default http://127.0.0.1:30000/v1)
    LLM_API_KEY            API key for the LLM              (default EMPTY)
    LLM_MODEL              model name                       (default qwen3.8-27b)
    STT_BASE_URL           OpenAI-compatible STT base URL   (default http://127.0.0.1:8100/v1)
    STT_API_KEY            API key for the STT              (default EMPTY)
    STT_MODEL              STT model name                   (default large-v3-turbo)
    TTS_BASE_URL           OpenAI-compatible TTS base URL   (default http://127.0.0.1:8880/v1)
    TTS_API_KEY            API key for the TTS              (default EMPTY)
    TTS_MODEL              TTS model name                   (default tts-1)
    TTS_VOICE              OpenAI-voice-name or Kokoro voice (default alloy)
    AGENT_SYSTEM_PROMPT    system prompt                    (default: short voice-assistant prompt)
"""

from __future__ import annotations

import os
import sys

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

AGENT_HOST = os.environ.get("AGENT_HOST", "0.0.0.0")
AGENT_PORT = int(os.environ.get("AGENT_PORT", "8765"))

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:30000/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "EMPTY")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.8-27b")

STT_BASE_URL = os.environ.get("STT_BASE_URL", "http://127.0.0.1:8100/v1")
STT_API_KEY = os.environ.get("STT_API_KEY", "EMPTY")
STT_MODEL = os.environ.get("STT_MODEL", "large-v3-turbo")

TTS_BASE_URL = os.environ.get("TTS_BASE_URL", "http://127.0.0.1:8880/v1")
TTS_API_KEY = os.environ.get("TTS_API_KEY", "EMPTY")
TTS_MODEL = os.environ.get("TTS_MODEL", "tts-1")
TTS_VOICE = os.environ.get("TTS_VOICE", "alloy")

SYSTEM_PROMPT = os.environ.get(
    "AGENT_SYSTEM_PROMPT",
    "You are a helpful voice assistant. "
    "Keep your answers short and conversational. "
    "Your output will be converted to audio, so avoid special characters, "
    "markdown, and code blocks.",
)

TRANSPORT_PARAMS = {
    "websocket": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        add_wav_header=False,
        serializer=ProtobufFrameSerializer(),
    ),
}


async def run_bot(transport, runner_args: RunnerArguments) -> None:
    """Build and run the STT -> LLM -> TTS pipeline for one connected client."""
    logger.info(
        "Starting speech agent: LLM=%s STT=%s TTS=%s/%s",
        LLM_MODEL, STT_MODEL, TTS_MODEL, TTS_VOICE,
    )

    stt = OpenAISTTService(
        api_key=STT_API_KEY,
        base_url=STT_BASE_URL,
        model=STT_MODEL,
    )
    llm = OpenAILLMService(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
    )
    tts = OpenAITTSService(
        api_key=TTS_API_KEY,
        base_url=TTS_BASE_URL,
        model=TTS_MODEL,
        voice=TTS_VOICE,
        sample_rate=24000,  # Kokoro's native sample rate
    )

    context = LLMContext([{"role": "system", "content": SYSTEM_PROMPT}])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    @worker.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info("Pipecat client ready.")
        context.add_message(
            {"role": "developer", "content": "Start by briefly introducing yourself."},
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Pipecat client connected.")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Pipecat client disconnected.")
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments) -> None:
    """Entry point discovered by the pipecat dev runner."""
    transport = await create_transport(runner_args, TRANSPORT_PARAMS)
    await run_bot(transport, runner_args)


def main() -> None:
    """Run the pipecat dev runner.

    The runner discovers `bot` on the `__main__` module. When launched via the
    console script (`speech-agent`), `bot` is not automatically on `__main__`,
    so we attach it explicitly.
    """
    import pipecat.runner.run as runner

    # Make `bot` discoverable by the runner when launched via the console
    # script (where the executed module is not this one).
    sys.modules["__main__"].bot = bot

    # The runner reads these globals for its default host/port.
    runner.RUNNER_HOST = AGENT_HOST
    runner.RUNNER_PORT = AGENT_PORT

    runner.main()


if __name__ == "__main__":
    main()
