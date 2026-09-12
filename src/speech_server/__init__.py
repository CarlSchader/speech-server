"""speech-server: self-hosted STT/TTS + voice agent, OpenAI-compatible.

Three services, one environment:

* ``speech-stt``   -- OpenAI-compatible ``/v1/audio/transcriptions`` (faster-whisper)
* ``speech-tts``   -- OpenAI-compatible ``/v1/audio/speech`` (Kokoro-82M via kokoro-onnx)
* ``speech-agent`` -- Pipecat voice agent chaining STT -> LLM -> TTS
"""

__version__ = "0.1.0"
