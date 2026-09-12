# NixOS module: services.speech
#
# Self-hosted speech stack: OpenAI-compatible STT (faster-whisper) and TTS
# (Kokoro-82M / kokoro-onnx) servers plus a Pipecat voice agent that chains
# STT -> LLM -> TTS against any OpenAI-compatible endpoint (e.g. the local
# SGLang server from sglang-nix). All three run from one pinned python
# environment (speech-server.packages.<system>.speechEnv) as hardened,
# auto-restarting systemd units on a shared `speech` system user.
#
# Optional `openWebUi` wiring emits the AUDIO_* env vars Open WebUI needs to
# use the local STT/TTS servers for dictation and read-aloud.
{
  config,
  lib,
  pkgs,
  ...
}: let
  cfg = config.services.speech;

  user = "speech";

  # Shared hardening for all three units: pure-CPU workloads, no GPU, no
  # privileged access. Models live in the per-service StateDirectory.
  hardening = {
    NoNewPrivileges = true;
    PrivateTmp = true;
    PrivateDevices = true;
    ProtectSystem = "strict";
    ProtectHome = true;
    ProtectKernelLogs = true;
    ProtectKernelModules = true;
    ProtectKernelTunables = true;
    ProtectControlGroups = true;
    ProtectHostname = true;
    RestrictRealtime = true;
    RestrictSUIDSGID = true;
    LockPersonality = true;
    SystemCallArchitectures = "native";
    RestrictAddressFamilies = [ "AF_INET" "AF_INET6" "AF_UNIX" "AF_NETLINK" ];
    UMask = "0077";
  };

  mkService =
    {
      unit,
      description,
      script,
      stateDir,
      memoryMax,
      env,
      environmentFile,
      after ? [ ],
      wants ? [ ],
      timeoutStartSec ? "30min",
    }:
    {
      systemd.services.${unit} = {
        description = description;
        wantedBy = [ "multi-user.target" ];
        wants = [ "network-online.target" ] ++ wants;
        after = [ "network-online.target" ] ++ after;
        path = [cfg.package];
        environment =
          {
            HOME = "/var/lib/${stateDir}";
            XDG_CACHE_HOME = "/var/lib/${stateDir}/.cache";
          }
          // env;
        serviceConfig = {
          ExecStart = "${cfg.package}/bin/${script}";
          User = user;
          Group = user;
          StateDirectory = stateDir;
          WorkingDirectory = "/var/lib/${stateDir}";

          Restart = "always";
          RestartSec = 5;
          # First start may download model weights (STT ~1.6 GB, TTS ~330 MB).
          TimeoutStartSec = timeoutStartSec;

          EnvironmentFile = lib.optional (environmentFile != null) environmentFile;
          MemoryMax = memoryMax;
        }
        // hardening;
      };
    };

  sttUnit =
    lib.mkIf cfg.stt.enable (
      mkService {
        unit = "speech-stt";
        description = "speech-server STT (OpenAI-compatible, faster-whisper ${cfg.stt.model})";
        script = "speech-stt";
        stateDir = "speech-stt";
        memoryMax = cfg.stt.memoryMax;
        environmentFile = cfg.stt.environmentFile;
        env =
          {
            STT_HOST = cfg.stt.host;
            STT_PORT = (toString cfg.stt.port);
            STT_MODEL = cfg.stt.model;
            STT_DEVICE = cfg.stt.device;
            STT_VAD_FILTER = (toString cfg.stt.vadFilter);
            # faster-whisper downloads model weights here on first start.
            HF_HOME = "/var/lib/speech-stt/huggingface";
          }
          // lib.optionalAttrs (cfg.stt.computeType != null) {
            STT_COMPUTE_TYPE = cfg.stt.computeType;
          }
          // cfg.stt.environment;
      }
    );

  ttsUnit =
    lib.mkIf cfg.tts.enable (
      mkService {
        unit = "speech-tts";
        description = "speech-server TTS (OpenAI-compatible, Kokoro-82M)";
        script = "speech-tts";
        stateDir = "speech-tts";
        memoryMax = cfg.tts.memoryMax;
        environmentFile = cfg.tts.environmentFile;
        timeoutStartSec = "20min";
        env =
          {
            TTS_HOST = cfg.tts.host;
            TTS_PORT = (toString cfg.tts.port);
            TTS_VOICE = cfg.tts.voice;
          }
          // cfg.tts.environment;
      }
    );

  agentUnit =
    lib.mkIf cfg.agent.enable (
      mkService {
        unit = "speech-agent";
        description = "speech-server voice agent (Pipecat: STT -> LLM -> TTS)";
        script = "speech-agent";
        stateDir = "speech-agent";
        memoryMax = cfg.agent.memoryMax;
        environmentFile = cfg.agent.environmentFile;
        # Start after the local backends it talks to.
        after = [ ]
          ++ lib.optionals cfg.stt.enable [ "speech-stt.service" ]
          ++ lib.optionals cfg.tts.enable [ "speech-tts.service" ];
        wants = [ ]
          ++ lib.optionals cfg.stt.enable [ "speech-stt.service" ]
          ++ lib.optionals cfg.tts.enable [ "speech-tts.service" ];
        env =
          {
            AGENT_HOST = cfg.agent.host;
            AGENT_PORT = (toString cfg.agent.port);
            LLM_BASE_URL = cfg.agent.llm.baseURL;
            LLM_API_KEY = cfg.agent.llm.apiKey;
            LLM_MODEL = cfg.agent.llm.model;
            STT_BASE_URL = "http://127.0.0.1:${toString cfg.stt.port}/v1";
            STT_MODEL = cfg.agent.sttModel;
            TTS_BASE_URL = "http://127.0.0.1:${toString cfg.tts.port}/v1";
            TTS_MODEL = "tts-1";
            TTS_VOICE = cfg.agent.voice;
          }
          // lib.optionalAttrs (cfg.agent.systemPrompt != null) {
            AGENT_SYSTEM_PROMPT = cfg.agent.systemPrompt;
          }
          // cfg.agent.environment;
      }
    );

in {
  options.services.speech = {
    enable = lib.mkEnableOption "the speech stack (enable individual services below)";

    package = lib.mkOption {
      type = lib.types.package;
      description = ''
        Python environment providing {file}`bin/speech-stt`,
        {file}`bin/speech-tts` and {file}`bin/speech-agent`. Normally
        `speech-server.packages.''${builtins.currentSystem}.speechEnv`,
        the environment pinned by this flake's uv.lock.
      '';
    };

    stt = {
      enable = lib.mkEnableOption "OpenAI-compatible STT server (faster-whisper)";

      host = lib.mkOption {
        type = lib.types.str;
        default = "127.0.0.1";
        description = "Address the STT API listens on.";
      };

      port = lib.mkOption {
        type = lib.types.port;
        default = 8100;
        description = "Port the STT API listens on.";
      };

      openFirewall = lib.mkOption {
        type = lib.types.bool;
        default = false;
        description = "Open the STT port in the firewall.";
      };

      model = lib.mkOption {
        type = lib.types.str;
        default = "large-v3-turbo";
        description = "faster-whisper model name (tiny/base/small/medium/large-v3/large-v3-turbo).";
      };

      device = lib.mkOption {
        type = lib.types.str;
        default = "auto";
        description = "Inference device: `auto`, `cpu` or `cuda`.";
      };

      computeType = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        example = "int8";
        description = "CTranslate2 compute type (null = auto: int8 on CPU, float16 on CUDA).";
      };

      vadFilter = lib.mkOption {
        type = lib.types.bool;
        default = true;
        description = "Apply the built-in VAD filter before transcription.";
      };

      memoryMax = lib.mkOption {
        type = lib.types.str;
        default = "8G";
        description = "systemd `MemoryMax=` for the unit.";
      };

      environment = lib.mkOption {
        type = lib.types.attrsOf lib.types.str;
        default = { };
        description = "Extra environment variables for the STT service.";
      };

      environmentFile = lib.mkOption {
        type = lib.types.nullOr lib.types.path;
        default = null;
        description = "Environment file for the STT service.";
      };
    };

    tts = {
      enable = lib.mkEnableOption "OpenAI-compatible TTS server (Kokoro-82M)";

      host = lib.mkOption {
        type = lib.types.str;
        default = "127.0.0.1";
        description = "Address the TTS API listens on.";
      };

      port = lib.mkOption {
        type = lib.types.port;
        default = 8880;
        description = "Port the TTS API listens on.";
      };

      openFirewall = lib.mkOption {
        type = lib.types.bool;
        default = false;
        description = "Open the TTS port in the firewall.";
      };

      voice = lib.mkOption {
        type = lib.types.str;
        default = "af_heart";
        example = "am_adam";
        description = ''
          Default Kokoro voice used for OpenAI voice ids and unknown voices.
          See `GET /v1/voices` for the full list (af_*, am_*, bf_*, bm_*, ...).
        '';
      };

      memoryMax = lib.mkOption {
        type = lib.types.str;
        default = "4G";
        description = "systemd `MemoryMax=` for the unit.";
      };

      environment = lib.mkOption {
        type = lib.types.attrsOf lib.types.str;
        default = { };
        description = ''
          Extra environment variables for the TTS service (e.g.
          `TTS_MODEL_FILE` / `TTS_VOICES_FILE` to pin local model files).
        '';
      };

      environmentFile = lib.mkOption {
        type = lib.types.nullOr lib.types.path;
        default = null;
        description = "Environment file for the TTS service.";
      };
    };

    agent = {
      enable = lib.mkEnableOption "Pipecat voice agent (browser UI, full STT -> LLM -> TTS loop)";

      host = lib.mkOption {
        type = lib.types.str;
        default = "0.0.0.0";
        description = "Address the agent UI/API listens on.";
      };

      port = lib.mkOption {
        type = lib.types.port;
        default = 8765;
        description = "Port the agent UI/API listens on.";
      };

      openFirewall = lib.mkOption {
        type = lib.types.bool;
        default = false;
        description = "Open the agent port in the firewall.";
      };

      llm = {
        baseURL = lib.mkOption {
          type = lib.types.str;
          default = "http://127.0.0.1:30000/v1";
          description = "OpenAI-compatible LLM base URL (e.g. the local SGLang server).";
        };
        apiKey = lib.mkOption {
          type = lib.types.str;
          default = "EMPTY";
          description = "API key for the LLM (arbitrary if the LLM is unauthenticated).";
        };
        model = lib.mkOption {
          type = lib.types.str;
          default = "qwen3.8-27b";
          description = "Model name the LLM serves.";
        };
      };

      voice = lib.mkOption {
        type = lib.types.str;
        default = "alloy";
        description = ''
          TTS voice for the agent: an OpenAI voice id (mapped by the TTS
          server) or a Kokoro voice name used directly.
        '';
      };

      sttModel = lib.mkOption {
        type = lib.types.str;
        default = "large-v3-turbo";
        description = "Model name sent to the local STT server (informational; the server serves one model).";
      };

      systemPrompt = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        description = "System prompt for the agent (null = built-in voice-assistant prompt).";
      };

      memoryMax = lib.mkOption {
        type = lib.types.str;
        default = "4G";
        description = "systemd `MemoryMax=` for the unit.";
      };

      environment = lib.mkOption {
        type = lib.types.attrsOf lib.types.str;
        default = { };
        description = "Extra environment variables for the agent service.";
      };

      environmentFile = lib.mkOption {
        type = lib.types.nullOr lib.types.path;
        default = null;
        description = "Environment file for the agent service.";
      };
    };

    openWebUi = {
      enable = lib.mkEnableOption "Open WebUI audio wiring (AUDIO_* environment variables)";

      sttModel = lib.mkOption {
        type = lib.types.str;
        default = "large-v3-turbo";
        defaultText = lib.literalExpression "services.speech.stt.model";
        description = "Model name Open WebUI sends to the STT server.";
      };

      ttsModel = lib.mkOption {
        type = lib.types.str;
        default = "tts-1";
        description = "Model name Open WebUI sends to the TTS server.";
      };

      voice = lib.mkOption {
        type = lib.types.str;
        default = "alloy";
        description = "OpenAI voice id Open WebUI sends for read-aloud (mapped to a Kokoro voice by the TTS server).";
      };

      apiKey = lib.mkOption {
        type = lib.types.str;
        default = "EMPTY";
        description = "API key sent to the local STT/TTS servers (arbitrary; they do not authenticate).";
      };

      env = lib.mkOption {
        type = lib.types.attrsOf lib.types.str;
        default =
          {
            AUDIO_TTS_ENGINE = "openai";
            AUDIO_TTS_OPENAI_API_BASE_URL = "http://127.0.0.1:${toString cfg.tts.port}/v1";
            AUDIO_TTS_OPENAI_API_KEY = cfg.openWebUi.apiKey;
            AUDIO_TTS_MODEL = cfg.openWebUi.ttsModel;
            AUDIO_TTS_VOICE = cfg.openWebUi.voice;
            AUDIO_STT_ENGINE = "openai";
            AUDIO_STT_OPENAI_API_BASE_URL = "http://127.0.0.1:${toString cfg.stt.port}/v1";
            AUDIO_STT_OPENAI_API_KEY = cfg.openWebUi.apiKey;
            AUDIO_STT_MODEL = cfg.openWebUi.sttModel;
          };
        defaultText = lib.literalExpression "see description";
        description = ''
          The computed `AUDIO_*` environment variables. When Open WebUI is
          enabled through `services.sglang.ui`, merge these there instead:
          `services.sglang.ui.environment = config.services.speech.openWebUi.env;`
        '';
      };
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion = cfg.openWebUi.enable -> cfg.stt.enable && cfg.tts.enable;
        message = "services.speech.openWebUi.enable requires services.speech.stt.enable and services.speech.tts.enable.";
      }
      {
        assertion = cfg.agent.enable -> cfg.stt.enable && cfg.tts.enable;
        message = "services.speech.agent.enable requires services.speech.stt.enable and services.speech.tts.enable.";
      }
    ];

    users.users.${user} = {
      isSystemUser = true;
      group = user;
      home = "/var/lib/speech";
      description = "speech-server service user";
    };
    users.groups.${user} = { };

    imports = [
      sttUnit
      ttsUnit
      agentUnit
    ];

    networking.firewall.allowedTCPPorts =
      lib.optional cfg.stt.openFirewall cfg.stt.port
      ++ lib.optional cfg.tts.openFirewall cfg.tts.port
      ++ lib.optional cfg.agent.openFirewall cfg.agent.port;

    services.open-webui.environment = lib.mkIf cfg.openWebUi.enable cfg.openWebUi.env;
  };
}
