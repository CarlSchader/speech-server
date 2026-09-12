# Research: self-hosted STT → LLM → TTS server, Open WebUI-compatible

**Goal.** A server (on this DGX Spark, `aarch64-linux`, 128 GB unified memory)
that hosts open-weight STT and TTS models, chains
**speech → text → LLM → speech**, plugs into the existing SGLang
OpenAI-compatible endpoint (`http://127.0.0.1:30000/v1`, Qwen3.8-27B NVFP4,
see `sglang-nix`), and hooks into the existing Open WebUI instance.
Everything must be open-source software with open weights, packaged as
NixOS modules + systemd units in the style of `sglang-nix` / `vllm-nix`.

All versions/licenses below verified 2026-09-12 against primary sources
(PyPI metadata, GitHub source, NixOS flake on this host).

---

## 1. Corrections to the earlier draft notes

| Earlier note | Finding |
| --- | --- |
| "Pipecat … BSL" | **Pipecat is BSD-2-Clause** (`LICENSE` in pipecat-ai/pipecat, "Copyright (c) 2024–2026, Daily", SPDX `BSD 2-Clause License`). Fully open source. |
| "Parakeet V3 … Apache 2.0" | Parakeet TDT 0.6B v2 model card says **CC-BY-4.0** (NeMo tooling is Apache-2.0). Attribution required; fine, but it is **offline/batch** (24-min segments), not a streaming ASR — wrong shape for the interactive loop. |
| "Kokoro-FastAPI … ~3 GB VRAM" | True, but that's the torch/transformers path. The **`kokoro-onnx`** package (MIT) runs the same 82M model on ONNX Runtime — CPU-only, no torch, no CUDA wheels. The right choice on aarch64. |
| "faster-whisper … ~250× real-time on a 4090" | That number is x86 CUDA. **CTranslate2's aarch64 PyPI wheels are CPU-only** — on this box faster-whisper would run on the 20-core Arm CPU. Still fast for turn-based STT, but not the "trivial GPU" story. |
| "LLM component just speaks OpenAI-compatible API" | Correct — and pipecat's OpenAI STT/LLM/TTS services all accept `base_url`, so all three can point at local servers. |

**Hardware note:** this host is a DGX Spark (GB10 / sm_121, aarch64), *not*
the 4090 workstation from the original notes. The recommended stack is
therefore tuned for Arm + unified memory: CPU inference for STT/TTS is the
default (cheap, reliable, no sm_121 wheel risk), with the LLM — the only
genuinely GPU-hungry part — already served by SGLang.

---

## 2. Component survey

### 2.1 STT (voice → text)

| Engine | License | Streaming | This host (aarch64) | Verdict |
| --- | --- | --- | --- | --- |
| **faster-whisper** (CTranslate2) | MIT | chunked / offline per utterance (pair with VAD) | CPU-only wheels; nixpkgs `python3Packages.faster-whisper` 1.2.1; OpenAI-compatible server is ~150 lines of FastAPI | **Primary.** Accuracy + permissive license + trivially pinnable via uv2nix. `large-v3-turbo` is the speed/accuracy sweet spot (~0.5 s per typical utterance on CPU). |
| **whisper.cpp** | MIT | same (chunked) | nixpkgs `whisper-cpp` 1.9.2; CUDA backend explicitly supports DGX Spark (v1.9.3: "MMVQ nwarps=8 … on DGX Spark") | Fast CUDA alternative; native server is *not* OpenAI-shaped (`/inference`, `/load`) so it needs a wrapper to plug into OWUI/pipecat. Kept as option, not default. |
| **Moonshine** (moonshine-ai) | MIT for streaming models (legacy non-English non-streaming variants are non-commercial) | true streaming, lowest input latency | no confirmed aarch64 GPU story; Python API still maturing | Strong latency pick; revisit once the aarch64 wheels are settled. |
| **Parakeet TDT 0.6B** | CC-BY-4.0 | no (offline, ≤ 24 min) | NeMo/torch aarch64 works (proven by sglang's torch) | Best-in-class batch WER; not for the interactive loop. |
| **Vosk** | Apache-2.0 | true streaming | CPU | Outdated accuracy; fallback only. |

**Endpointing:** Silero VAD (MIT). Pipecat bundles the ONNX model *inside*
the pipecat wheel (`pipecat/audio/vad/data/silero_vad.onnx`) — no runtime
download.

### 2.2 TTS (text → voice)

| Engine | License | Shape | Notes |
| --- | --- | --- | --- |
| **Kokoro-82M via kokoro-onnx** | model Apache-2.0, lib MIT | ONNX Runtime, CPU, 24 kHz | **Primary.** ~330 MB (onnx + voices). Pipecat ships `KokoroTTSService` on top of it. A small FastAPI wrapper gives the OpenAI-compatible `/v1/audio/speech`. |
| **Chatterbox (Resemble AI)** | MIT | torch | 0.5B, zero-shot cloning from ~10 s of audio, 23+ languages (v3). Higher quality; needs torch aarch64 wheels (same risk class as sglang, but more moving parts). |
| **Piper** | MIT | C++, CPU | nixpkgs `piper-tts` 1.8.0. Very light, lower quality. Good fallback / minimal target. |
| Kokoro via torch (hexgrad/Kokoro, kokoro-fastapi) | Apache-2.0 | torch + transformers | What the original notes referenced; the ONNX path is strictly better here. |
| F5-TTS | **CC-BY-NC-4.0** | — | Excluded (non-commercial). |
| XTTS v2 | **CPML** (Coqui) | — | Excluded (non-commercial-leaning license). |
| GPT-SoVITS | mixed | — | Excluded (weight provenance/license murky). |

### 2.3 Glue (STT → LLM → TTS loop)

| Framework | License | Fit |
| --- | --- | --- |
| **Pipecat (pipecat-ai)** | **BSD-2-Clause** | **Primary.** Purpose-built pipeline: VAD turn-taking, barge-in/interruption, streaming tokens straight into TTS. 1.x has local, no-cloud services: `WhisperSTTService` (faster-whisper), `KokoroTTSService` (kokoro-onnx), `OpenAILLMService` (any OpenAI-compatible endpoint → SGLang), `SileroVADAnalyzer`. A runner API (`pipecat.runner.run:main`) serves a FastAPI/WebSocket app with a prebuilt browser client (`pipecat-ai-prebuilt`). |
| LiveKit Agents | Apache-2.0 (framework & server) | The production-grade WebRTC alternative for *remote* users (browsers/phones over the network, TURN, rooms). More moving parts (LiveKit server). Revisit if the agent must be reachable off-LAN. |
| RealtimeTTS (KoljaB) | MIT | LLM-token-stream → TTS only; you'd own STT, VAD, transport, turn-taking. More code to maintain for the same result. |
| DIY FastAPI loop | — | Viable, but Pipecat's interruption handling is the hard part to get right; not worth re-inventing. |

**Pipecat 1.x API notes (verified against pipecat-ai 1.10.0 wheel):**
- `OpenAILLMService`, `OpenAISTTService`, `OpenAITTSService` all accept
  `base_url` (verified in `pipecat/services/openai/{llm,stt,tts}.py`), so
  pointing them at `http://127.0.0.1:30000/v1` (SGLang) and at the local
  STT/TTS servers is first-class.
- Modern wiring (per `examples/voice/voice-openai-http.py` in
  pipecat-ai/pipecat, main branch):
  `Pipeline([transport.input(), stt, user_aggregator, llm, tts,
  transport.output(), assistant_aggregator])` where the aggregator pair
  comes from `LLMContextAggregatorPair(context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()))`.
- The websocket transport is `FastAPIWebsocketParams`
  (+ `ProtobufFrameSerializer`); the dev runner serves a prebuilt browser
  client UI, selected with `-t websocket`.
- Extras needed: `pipecat-ai[websocket,runner]` (fastapi, uvicorn,
  `pipecat-ai-prebuilt`). OpenAI SDK, onnxruntime (for Silero VAD),
  faster-whisper and kokoro-onnx are separate extras
  (`[openai]` is empty — openai is a core dep; `[whisper]`, `[kokoro]`).

### 2.4 LLM

No changes to SGLang: it already serves streaming OpenAI-compatible
`/v1/chat/completions`. Pipecat's `OpenAILLMService` consumes it with
`base_url=http://127.0.0.1:30000/v1`, `api_key` arbitrary (SGLang is
unauthenticated on the LAN). First-token latency (~0.3–1 s on the
Qwen3.8-27B preset) dominates end-to-end latency; STT and TTS are each
well under a second.

---

## 3. Open WebUI integration

Verified against Open WebUI source (`backend/open_webui/routers/audio.py`,
main branch) and current docs:

- **TTS** — per-message "read aloud". OWUI does
  `POST {AUDIO_TTS_OPENAI_API_BASE_URL}/audio/speech` with JSON
  `{model, input, voice?, ...}` (model/voice from its audio config; it
  merges `audio.tts.openai.params`). The response body is audio; OWUI
  transcodes non-`audio/mpeg` bodies to MP3 (via pydub/ffmpeg in its own
  package). ⇒ serve **MP3** to avoid the transcode.
- **STT** — per-message dictation. OWUI does
  `POST {AUDIO_STT_OPENAI_API_BASE_URL}/audio/transcriptions` as multipart
  form (`file`, `model`, optional `language`), OpenAI Whisper-shaped.
  It may also send the JSON variant (`audio.stt.openai.api_request_format`)
  with base64 `input_audio` — support both.
- Relevant env vars (set on the `open-webui` service, merging with the
  vars `sglang-nix` already sets):
  ```
  AUDIO_TTS_ENGINE=openai
  AUDIO_TTS_OPENAI_API_BASE_URL=http://127.0.0.1:8880/v1
  AUDIO_TTS_OPENAI_API_KEY=EMPTY
  AUDIO_TTS_MODEL=tts-1
  AUDIO_TTS_VOICE=alloy          # our server maps unknown voices to its default
  AUDIO_STT_ENGINE=openai
  AUDIO_STT_OPENAI_API_BASE_URL=http://127.0.0.1:8100/v1
  AUDIO_STT_OPENAI_API_KEY=EMPTY
  AUDIO_STT_MODEL=whisper-large-v3-turbo
  ```
- **Limitation (inherent, not a bug):** OWUI audio is request/response —
  dictate a message, hear the reply after it's generated. It is *not*
  full-duplex conversation. That experience lives in the Pipecat agent
  app instead; both share the same STT/TTS/LLM backends, so voices and
  quality stay consistent across both interfaces.

---

## 4. Target architecture

```
                        ┌────────────────────────────────────────────┐
                        │ Open WebUI (:8080, existing, sglang-nix ui)│
                        │  dictation ─┐        ┌─ read-aloud         │
                        └─────────────┼────────┼─────────────────────┘
                                      ▼        ▼
  mic ──▶ Pipecat agent (:8765, browser UI)     │
          SileroVAD → OpenAISTT → OpenAILLM     │
          → OpenAITTS → speaker (barge-in)      │
                │          │          │         │
                ▼          ▼          ▼         ▼
   ┌───────────────────────────────────────────────────────┐
   │  speech-server (this repo)                            │
   │  services.speech.stt   :8100  OpenAI /v1/audio/transcriptions  │
   │  services.speech.tts   :8880  OpenAI /v1/audio/speech (mp3)     │
   │  services.speech.agent :8765  pipecat runner + prebuilt client  │
   └───────────────────────────────────────────────────────┘
                │                      │
                └──────────┬───────────┘
                           ▼
              SGLang :30000/v1 (Qwen3.8-27B NVFP4, existing)
```

**Services** (each its own systemd unit, same hardening pattern as
`sglang-nix`):

1. **`speech-stt`** — FastAPI + faster-whisper, OpenAI-compatible
   `/v1/audio/transcriptions` (+ `/v1/models`, `/health`). Model
   `large-v3-turbo` by default; auto-download to a state dir on first
   start (same pattern as sglang's weight cache).
2. **`speech-tts`** — FastAPI + kokoro-onnx, OpenAI-compatible
   `/v1/audio/speech` (MP3/WAV/PCM out; voice aliases `alloy`/`echo`/…
   → Kokoro voices so OWUI "just works"; unknown voice falls back to the
   default with a log warning).
3. **`speech-agent`** — pipecat runner: browser UI at `:8765`
   (`-t websocket`), full STT→LLM→TTS loop with barge-in; LLM base URL
   defaults to the local SGLang when `services.sglang` is enabled.

**Nix packaging** (mirrors sglang-nix exactly):
- One `pyproject.toml` + `uv.lock` at the repo root; PEP 735 dependency
  groups `stt`/`tts`/`agent`; uv2nix `sourcePreference = "wheel"` keeps
  the no-source-builds promise. All chosen deps have aarch64 wheels
  (verified on PyPI: ctranslate2, onnxruntime, phonemizer, soundfile,
  numpy, soxr, numba, av, Pillow, pydantic-core, aiohttp, websockets,
  …). No torch in any env.
- One venv (`speechEnv`) with three console scripts
  (`speech-stt`, `speech-tts`, `speech-agent`); the NixOS module exposes
  `services.speech.{stt,tts,agent,openWebUi}` with per-service `enable`,
  one system user, per-service `StateDirectory` + `HF_HOME` so models
  survive updates.
- Preset `services.speech` DGX-Spark defaults: everything on,
  localhost binding, OWUI wiring on.
- Checks: venv import check (all key modules import, no GPU, no model
  downloads) + `nixosSystem` module-eval check that pins the rendered
  `ExecStart`/firewall/OWUI env, mirroring `sglangEnvImport` /
  `sglangModuleEval`.

**Memory budget:** SGLang holds ~64 GB (mem-fraction 0.50, `MemoryMax=100G`).
Speech stack: STT ≈ 1–2 GB, TTS ≈ 0.5 GB, agent ≈ 0.5 GB → **≤ 3–4 GB**,
per-unit `MemoryMax` as a backstop. No GPU pressure: STT/TTS are CPU.

---

## 5. Risks / watch items

1. **CTranslate2 aarch64 = CPU-only wheels.** Fine for v1; if STT latency
   ever matters, swap the STT engine to whisper.cpp CUDA (nixpkgs-native,
   DGX Spark support in 1.9.3) behind the same OpenAI endpoint.
2. **kokoro-onnx / phonemizer espeak-ng runtime:** `espeakng-loader`
   bundles the espeak-ng library + data; validate at import/smoke-test
   time on aarch64.
3. **OWUI audio env var names** (`AUDIO_TTS_*` / `AUDIO_STT_*`) move with
   OWUI releases; the module sets them via `services.open-webui.environment`
   so they can be adjusted without touching the service unit. Re-verify
   against the pinned OWUI version during rollout.
4. **Pipecat 1.x is young** (API churn between 0.x and 1.x was large).
   The env is pinned via `uv.lock`; upgrade deliberately, re-reading the
   current `examples/voice/*` first.
5. **Voice naming:** OWUI sends OpenAI voice ids (`alloy`, …). The TTS
   server aliases them to Kokoro voices (`af_heart`, …) and falls back to
   the configured default for anything unknown.
6. Excluded on license: F5-TTS (CC-BY-NC), XTTS v2 (CPML), GPT-SoVITS
   (mixed). Moonshine legacy non-English models (non-commercial) — only
   the MIT streaming variants are usable.

## 6. Key sources

- `~/code/sglang-nix` (module/env/hardening pattern, OWUI env wiring),
  `~/code/nix-config/.../dgx-spark` (host setup), `~/code/vllm-nix`.
- Pipecat: `github.com/pipecat-ai/pipecat` (LICENSE = BSD-2-Clause;
  `examples/voice/voice-openai-http.py`), pipecat-ai 1.10.0 wheel
  (services/openai/{llm,stt,tts}.py `base_url`; `audio/vad/data/silero_vad.onnx`
  bundled; `serializers/protobuf.py`; `runner/`), `github.com/pipecat-ai/pipecat-examples`
  (websocket runner usage), PyPI metadata for extras.
- Kokoro: `github.com/thewh1teagle/kokoro-onnx` (0.6.1, MIT; `Kokoro(model_path,
  voices_path)` + `create_stream`; 24 kHz; model files from
  `kokoro-onnx/releases/model-files-v1.0`), `huggingface.co/hexgrad/Kokoro-82M`
  (Apache-2.0).
- STT: `github.com/SYSTRAN/faster-whisper` (MIT),
  `github.com/ggml-org/whisper.cpp` (MIT; 1.9.3 DGX Spark CUDA note;
  nixpkgs `whisper-cpp` 1.9.2), `github.com/moonshine-ai/moonshine` (MIT
  streaming; legacy variants non-commercial),
  `huggingface.co/nvidia/parakeet-tdt-0.6b-v2` (CC-BY-4.0, offline).
- Excluded: `huggingface.co/coqui/XTTS-v2` (CPML), F5-TTS (CC-BY-NC-4.0).
- Open WebUI: `github.com/open-webui/open-webui` `backend/open_webui/routers/audio.py`
  (`_tts_openai`, `_transcribe_openai`, transcode-to-mp3 behaviour),
  `docs.openwebui.com` audio docs.
- Nix: nixpkgs-unstable on this host (`whisper-cpp`, `piper-tts`,
  `whisper-ctranslate2`, `python3Packages.faster-whisper`),
  `github.com/pyproject-nix/uv2nix` + `pyproject.nix` (`loadWorkspace`,
  `mkVirtualEnv` spec form `{ pkg = [ groups... ]; }`).
