# CUDA on an RTX 4090 and remote vLLM

Use `packages.x86_64-linux.speechEnvCuda` instead of `speechEnv` and enable
CUDA explicitly. CPU remains the default; the DGX preset is unchanged.

```nix
services.speech = {
  enable = true;
  package = inputs.speech-server.packages.x86_64-linux.speechEnvCuda;
  stt = { enable = true; device = "cuda"; computeType = "int8_float16"; };
  tts = { enable = true; device = "cuda"; cpuThreads = 4; };
  agent = {
    enable = true;
    host = "127.0.0.1";
    port = 8766;
    llm.baseURL = "https://llm.carlschader.com/v1";
    llm.model = "qwen3.8-27b";
    environment.LLM_EXTRA_BODY = builtins.toJSON {
      chat_template_kwargs.enable_thinking = false;
    };
  };
};
```

The host must provide working NVIDIA drivers. CUDA units load CUDA 12/cuDNN 9
libraries and expose only NVIDIA device classes; CPU units retain PrivateDevices.
The ONNX GPU wheel is pinned by SHA256; the unused TensorRT provider is removed.
The x86_64 Numba wheel's TBB dependency is explicit. Upstream CPU wheels and
lock-file dependency versions otherwise remain unchanged. `nix/package.nix`
can also be imported with downstream `pkgs` and the flake's uv2nix inputs.

For a reverse-proxied agent, override ExecStart to append
`--transport websocket --proxy voice.example.com`, enable WebSocket forwarding,
set `PIPECAT_ALLOWED_ORIGINS=https://voice.example.com` and
`PIPECAT_WEBSOCKET_AUTH=token`. A pinned Pipecat patch makes `/start` advertise
the public WSS URL rather than its private bind address. The agent supports
WebSocket transport only. The runner's session token is not user authentication;
keep the UI/APIs on a trusted private network or add real access control.

Open WebUI on another host must use remote STT/TTS URLs, not the localhost
values computed by `openWebUi.env`. Its persistent Admin Audio settings may
override environment defaults. Do not enable this module's automatic WebUI
instance when WebUI is already running elsewhere.

## Benchmark evidence

Recorded 2026-10-26 on Ryzen 7900X, RTX4090, 32 GiB RAM, driver 595.104.02.
Synthetic English short question and 21.74-second passage, one first call plus
five warm calls, sequential configurations on a live workstation.

| Long passage warm median | CPU 4 threads | CPU 12 threads | CUDA |
|---|---:|---:|---:|
| STT | 3.184 s | 2.470 s | 0.243 s INT8/FP16; 0.254 s FP16 |
| TTS | 3.504 s | 2.397 s | 0.626 s |

Transcripts were identical across tested configurations. No noisy/multilingual
accuracy or subjective TTS quality claim is made. CUDA STT sampled process VRAM:
1588 MiB INT8/FP16 vs 2548 MiB FP16. CUDA TTS: 2492 MiB. CPU and GPU runtimes
load models in system RAM too; these numbers are not system memory limits.

Coexistence used a **separate, bounded** sd-cpp instance with existing
Qwen-Image-2.1 weights on CUDA, no CPU offload, tiled VAE, 1024²/20 steps.
Three images/configuration while continuously alternating short/long speech pairs:

| Speech configuration | Short STT + TTS medians | Image median | Peak total GPU |
|---|---:|---:|---:|
| None | — | 26.28 s | 14452 MiB |
| CPU/CPU 12 threads | 2.292 + 0.438 s | 26.65 s | 14257 MiB |
| CUDA/CPU | 0.514 + 0.431 s | 28.25 s | 15850 MiB |
| CUDA/CUDA | 0.513 + 0.266 s | 32.39 s | 17954 MiB |

Dual CUDA favors voice latency; mixed CUDA/CPU favors images. The production
CPU-offload image baseline exhausted host RAM and was aborted; it is not a valid
performance baseline. A missing-driver-path experiment fell back to CPU and hit
its 6 GiB cgroup limit. Both failures are recorded, not silently excluded.
Never start an experimental image server alongside the production service.
Stop production via systemd first; cap test RAM/swap and stop test processes
before restarting production. Other apps/gaming, larger images and many users
can invalidate these capacity measurements.

Scripts: `benchmarks/benchmark.py` and `benchmarks/coexist.py`.
Raw timings, outputs, model hashes and Whisper snapshot:
[`tower-2026-10-26.json`](../benchmarks/results/tower-2026-10-26.json).
Set CUDA runtime library paths (service environments do this automatically).
For isolated CPU tests use `speechEnv`, for CUDA use `speechEnvCuda`:

```sh
python benchmarks/benchmark.py --cache "$HOME/.cache/speech-benchmark" \
  --task prepare --output /tmp/prepare.json
python benchmarks/benchmark.py --cache "$HOME/.cache/speech-benchmark" \
  --task stt --device cuda --compute int8_float16 --output /tmp/stt.json
python benchmarks/benchmark.py --cache "$HOME/.cache/speech-benchmark" \
  --task tts --device cuda --output /tmp/tts.json
```

`tests/test_api.py` is model-free. `tests/smoke_live.py` is opt-in: it starts
only its own loopback subprocesses, uses the cached synthetic fixtures and
models, and calls the real DGX vLLM endpoint. Set `SPEECH_ENV` to the built
CUDA environment and `SPEECH_BENCH_CACHE` to the fixture cache. It checks STT
verbose JSON, vLLM, TTS PCM/WAV/MP3, public WSS URL generation, CORS behavior
and a token-authenticated WebSocket session that returns synthesized audio.
Real browser microphone and barge-in acceptance still require manual testing.
