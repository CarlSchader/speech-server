# speech-server

Self-hosted **STT → LLM → TTS** voice stack for NixOS, in the style of
[`sglang-nix`](https://github.com/): pinned `uv.lock` → uv2nix wheel-only
environment → hardened systemd units, all open source + open weights.

Three services, one python environment (`speechEnv`):

| Unit | Port | What it is | Backend |
| --- | --- | --- | --- |
| `speech-stt` | 8100 | OpenAI-compatible `POST /v1/audio/transcriptions` | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (MIT, `large-v3-turbo`, CTranslate2 on CPU) |
| `speech-tts` | 8880 | OpenAI-compatible `POST /v1/audio/speech` (mp3/wav/pcm) | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) via [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) (Apache-2.0 model, MIT lib, ONNX Runtime on CPU) |
| `speech-agent` | 8765 | Real-time voice agent (Pipecat runner + browser UI): mic → Silero VAD → STT → **SGLang LLM** → TTS → speaker, with barge-in | [pipecat-ai](https://github.com/pipecat-ai/pipecat) (BSD-2-Clause), all OpenAI-compatible services pointed at the local servers |

Plus `services.speech.openWebUi` which computes the `AUDIO_*` environment
variables Open WebUI needs for dictation (STT) and read-aloud (TTS) against
those servers.

Everything is CPU-bound on the machine: the LLM (the only GPU-hungry part) is
the existing SGLang server. On a DGX Spark the speech stack budgets ~5–6 GB of
the 128 GB unified memory.

See [docs/usage.md](docs/usage.md) for deployment and
[RESEARCH.md](RESEARCH.md) for the full component survey and reasoning.

## Layout

```
flake.nix            flake: packages (speechEnv), nixosModules, checks, dev shell
uv.lock              pinned wheel resolution (aarch64 + x86_64, no torch)
pyproject.toml       project + dependency groups: stt / tts / agent
src/speech_server/   the three services (stt.py, tts.py, agent.py)
nix/env.nix          uv2nix workspace -> single venv with all three console scripts
modules/speech.nix   NixOS module: services.speech.{stt,tts,agent,openWebUi}
modules/dgx-spark.nix  DGX Spark preset (enable everything, LLM on :30000)
```

## Quick start (nix)

```sh
# Build the python env (wheel-only; fails loudly if a dep lacks an aarch64 wheel)
nix build .#packages.aarch64-linux.speechEnv

# Run checks: venv import smoke test + NixOS module evaluation
nix flake check

# Dev shell (uv for local iteration: uv run speech-stt, ...)
nix develop
```

## NixOS module (DGX Spark example)

```nix
{
  imports = [ ./speech-server/modules/speech.nix ./speech-server/modules/dgx-spark.nix ];

  services.speech = {
    enable = true;
    package = (import ./speech-server).packages.aarch64-linux.speechEnv;
    # DGX preset turns on stt/tts/agent; LLM defaults to 127.0.0.1:30000/v1
    # (services.sglang with the dgx-spark-qwen38 preset).
    agent.openFirewall = true;
  };

  # If Open WebUI is enabled by services.sglang.ui (as on this host), wire
  # its audio to the local servers:
  services.sglang.ui.environment = config.services.speech.openWebUi.env;
}
```

Without the preset, enable services individually — see
[docs/usage.md](docs/usage.md).
