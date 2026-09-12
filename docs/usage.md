# Usage

## Deployment on the DGX Spark

The host already runs SGLang (`services.sglang` with the
`dgx-spark-qwen38` preset from `sglang-nix`) and Open WebUI
(`services.sglang.ui`). The speech stack slots in next to it:

```nix
# in nix-config (dgx-spark)
{
  imports = [
    (fetchTarball (pkgs.fetchFromGitHub { ... })) # or add the repo as a flake input
  ];
}
```

Concretely, if the repo is available at `~/code/speech-server` on the host:

```nix
# modules or configuration.nix
let
  speechServer = builtins.getFlake "/home/carl/code/speech-server";
in {
  imports = [
    speechServer.nixosModules.speech
    speechServer.nixosModules.dgx-spark
  ];

  services.speech = {
    enable = true;
    package = speechServer.packages.aarch64-linux.speechEnv;
    agent.openFirewall = true; # laptop browser can open :8765 on the LAN
  };

  # Wire Open WebUI's dictation + read-aloud to the local servers
  # (OWUI here is enabled by services.sglang.ui, which owns its env):
  services.sglang.ui.environment = config.services.speech.openWebUi.env;
}
```

If your host enables Open WebUI through the generic `services.open-webui`
module instead, use `services.speech.openWebUi.enable = true;` — it sets
`services.open-webui.environment` itself.

Then `nixos-rebuild switch`. First boot downloads the models:
STT ~1.6 GB (faster-whisper `large-v3-turbo` → `~/.cache` under
`/var/lib/speech-stt`), TTS ~330 MB (`/var/lib/speech-tts/models/`).
The units set `TimeoutStartSec=30min` for exactly this.

## What you get

- **Open WebUI** — no new UI. The mic/dictation button in any chat
  transcribes via `speech-stt`; "read aloud" (the speaker icon) synthesises
  the reply via `speech-tts` (Kokoro `af_heart` voice; OpenAI voice ids like
  `alloy` are mapped in the TTS server). Request/response, per message.
- **Voice agent** — `http://<spark>:8765` serves Pipecat's prebuilt browser
  UI: full-duplex-ish conversation with barge-in (interrupting the agent stops
  its speech). The loop is Silero VAD → `speech-stt` → SGLang (`/v1`) →
  `speech-tts`.
- **Raw endpoints** — the two servers are plain OpenAI-compatible:
  `curl -F file=@audio.mp3 http://127.0.0.1:8100/v1/audio/transcriptions`
  and `curl -H 'Content-Type: application/json' -d '{"input":"hi","model":"tts-1"}'
  http://127.0.0.1:8880/v1/audio/speech` (default response is MP3;
  `response_format` = `wav`|`pcm`|`mp3`).

## Configuration knobs (all `mkDefault` in the DGX preset)

| Option | Default | Meaning |
| --- | --- | --- |
| `services.speech.stt.model` | `large-v3-turbo` | faster-whisper model |
| `services.speech.stt.device` / `.computeType` | `auto` / `int8`-on-CPU | CTranslate2 backend |
| `services.speech.tts.voice` | `af_heart` | fallback voice for unknown/OpenAI ids |
| `services.speech.agent.llm.{baseURL,model}` | `http://127.0.0.1:30000/v1`, `qwen3.8-27b` | the LLM backend |
| `services.speech.agent.voice` | `alloy` | TTS voice for the agent UI |
| `services.speech.agent.systemPrompt` | built-in | agent persona |
| `services.speech.*.memoryMax` | 8G / 4G / 4G | cgroup backstops |

TTS voices: `GET http://127.0.0.1:8880/v1/voices`. OpenAI ids
(`alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`, `coral`, `sage`) map to
Kokoro voices; anything else falls back to the default voice with a log
warning.

## Day-2

```sh
# Tail / inspect
journalctl -u speech-stt -u speech-tts -u speech-agent
systemctl restart speech-tts
curl -s localhost:8100/health
curl -s localhost:8880/health

# Re-pin dependencies (uv), then rebuild:
uv lock
nix build .#packages.aarch64-linux.speechEnv
nix flake check
```

## Limitations / risks

- **STT is chunked, not token-streaming**: transcription happens per
  utterance (Silero VAD endpoint). Fine for turn-taking; expect a brief pause
  after you finish speaking before the LLM starts.
- **CTranslate2 aarch64 wheels are CPU-only** — STT runs on the Arm CPU.
  `large-v3-turbo` is ~real-time on 20 cores; swap `stt.model` for
  `large-v3` (slower, more accurate) or a smaller model as needed. If CPU
  latency ever bites, the STT endpoint can be re-backed by whisper.cpp CUDA
  (DGX Spark supported) behind the same OpenAI-compatible surface.
- **Kokoro is English-centric**: quality is best for the `af_*`/`am_*`
  voices; other languages (`em_*` Spanish, `ef_*` French, `zf_*` Mandarin,
  `jf_*` Japanese, ...) work but are less natural. Multilingual alternatives
  that are in scope: Chatterbox (MIT, zero-shot cloning) or Piper (MIT, light).
- **OWUI audio is request/response** (dictate a message, hear the reply);
  the interactive loop is the Pipecat agent UI. Both share the same STT/TTS
  servers, so voice and quality stay consistent.
- **OpenAI voice ids** are cosmetic: they map to Kokoro voices server-side;
  OWUI has no way to list our actual voices.
- License-excluded by design: XTTS v2 (CPML), F5-TTS (CC-BY-NC-4.0),
  GPT-SoVITS (mixed), Moonshine legacy non-English models (non-commercial).
