# Preset: speech stack for the DGX Spark host (aarch64-linux).
#
# Everything runs on the 20-core Arm CPU — SGLang owns the GPU for the LLM.
# STT/TTS bind to loopback; only the agent UI is exposed to the LAN so a
# browser on another machine can reach it. LLM defaults match
# services.sglang with the dgx-spark-qwen38 preset.
#
# Usage: set `services.speech.enable = true;` in a module imported after
# this preset; all values here are mkDefault and stay overridable.
#
# Open WebUI: if the OWUI instance is the one enabled by services.sglang.ui
# (as on this host), wire the audio env there instead of openWebUi.enable:
#   services.sglang.ui.environment = config.services.speech.openWebUi.env;
{
  config,
  lib,
  ...
}: {
  config = lib.mkIf config.services.speech.enable {
    services.speech = {
      stt.enable = lib.mkDefault true;
      stt.model = lib.mkDefault "large-v3-turbo";
      stt.device = lib.mkDefault "auto";
      stt.memoryMax = lib.mkDefault "8G";

      tts.enable = lib.mkDefault true;
      tts.voice = lib.mkDefault "af_heart";
      tts.memoryMax = lib.mkDefault "4G";

      agent.enable = lib.mkDefault true;
      agent.host = lib.mkDefault "0.0.0.0";
      agent.openFirewall = lib.mkDefault true;
      agent.memoryMax = lib.mkDefault "4G";
      agent.llm = {
        baseURL = lib.mkDefault "http://127.0.0.1:30000/v1";
        model = lib.mkDefault "qwen3.8-27b";
      };

      openWebUi.enable = lib.mkDefault false; # OWUI here is owned by services.sglang.ui
    };
  };
}
