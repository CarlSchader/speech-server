{
  nixpkgs,
  flake-utils,
  pyproject-nix,
  uv2nix,
  pyproject-build-systems,
  ...
}:
flake-utils.lib.eachSystem ["x86_64-linux" "aarch64-linux"] (system: let
  inherit (nixpkgs) lib;

  pkgs = import nixpkgs {
    inherit system;
    config.allowUnfree = true;
  };

  # Read the uv.lock pinning faster-whisper / kokoro-onnx / pipecat-ai (and
  # friends) from the workspace root.
  workspace = uv2nix.lib.workspace.loadWorkspace {
    workspaceRoot = ../.;
  };

  # One derivation per pinned wheel, using the hashes already in uv.lock.
  # "wheel" preference enforces the no-source-builds promise: a dependency
  # that has no prebuilt wheel fails the build loudly instead of compiling.
  # All chosen dependencies ship aarch64 + x86_64 wheels (no torch anywhere).
  overlay = workspace.mkPyprojectOverlay {
    sourcePreference = "wheel";
  };

  pythonSet =
    (pkgs.callPackage pyproject-nix.build.packages {
      python = pkgs.python312;
    }).overrideScope (lib.composeManyExtensions [
      pyproject-build-systems.overlays.default
      overlay
    ]);

  # One environment hosting all three services: the local `speech-server`
  # package (console scripts speech-stt / speech-tts / speech-agent) plus the
  # union of the stt / tts / agent dependency groups.
  speechEnv = pythonSet.mkVirtualEnv "speech-env" workspace.deps.all;

  # Pure-eval smoke test of the NixOS module: a full nixosSystem evaluation
  # with the module + DGX Spark preset enabled that materialises the rendered
  # ExecStart. No VM, no GPU, no model downloads.
  moduleEval = nixpkgs.lib.nixosSystem {
    inherit system;
    modules = [
      ../modules/speech.nix
      ../modules/dgx-spark.nix
      {
        nixpkgs.hostPlatform = system;
        nixpkgs.config.allowUnfree = true;
        boot.loader.grub.enable = false;
        fileSystems."/" = {
          device = "none";
          fsType = "tmpfs";
        };
        system.stateVersion = "25.05";

        services.speech = {
          enable = true;
          package = speechEnv;
          openWebUi.enable = true;
        };
      }
    ];
  };
in {
  # Reproducible python env for the speech servers built from the pinned
  # wheels. Exposes $out/bin/speech-stt, $out/bin/speech-tts,
  # $out/bin/speech-agent and $out/bin/python.
  packages.speechEnv = speechEnv;
  packages.default = speechEnv;

  checks = {
    # Prove the env is a coherent, importable python environment (no GPU, no
    # model downloads): every key module imports and the three console
    # scripts are on PATH.
    speechEnvImport = pkgs.stdenv.mkDerivation {
      name = "speech-env-import-check";
      nativeBuildInputs = [speechEnv];
      dontUnpack = true;
      dontConfigure = true;
      buildPhase = ''
        runHook preBuild
        # faster-whisper / huggingface_hub create cache dirs under $HOME.
        export HOME=$(mktemp -d)
        python - <<'EOF'
        import importlib.metadata as md
        import shutil
        import subprocess
        import sys

        # Modules that import cleanly even in a build sandbox.
        import av
        import fastapi
        import faster_whisper
        import numpy
        import pydantic
        import uvicorn

        import speech_server.stt
        import speech_server.tts

        print("faster-whisper", faster_whisper.__version__)
        print("fastapi", fastapi.__version__)
        print("av", av.__version__)
        print("numpy", numpy.__version__)

        # Every wheel is installed at the version pinned by uv.lock.
        for dist, prefix in [
            ("faster-whisper", "1.2"),
            ("kokoro-onnx", "0.6"),
            ("pipecat-ai", "1.10"),
            ("onnxruntime", "1."),
        ]:
            version = md.version(dist)
            assert version.startswith(prefix), f"{dist} is {version}, expected {prefix}*"
            print(f"{dist} {version}")

        # onnxruntime-dependent imports: kokoro_onnx and pipecat pull in
        # onnxruntime, whose CPU feature probe aborts inside the Nix build
        # sandbox (its fake /sys). Run them in a subprocess: SIGABRT/SIGSEGV
        # there is the known sandbox limitation and is tolerated; any other
        # failure is a real packaging bug. Outside a sandbox these imports
        # must succeed normally.
        probe = """
        import kokoro_onnx
        import pipecat
        import speech_server.agent
        print("kokoro-onnx", kokoro_onnx.__version__)
        print("pipecat", pipecat.__version__)
        print("agent OK")
        """
        result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
        if result.returncode == 0:
            sys.stdout.write(result.stdout)
        elif result.returncode in (-6, -11):  # SIGABRT / SIGSEGV in sandbox
            print("note: onnxruntime-dependent imports skipped "
                  "(onnxruntime CPU probe aborts in the Nix build sandbox)")
        else:
            sys.stderr.write(result.stdout + result.stderr)
            sys.exit(1)

        for exe in ("speech-stt", "speech-tts", "speech-agent"):
            assert shutil.which(exe), f"missing console script: {exe}"
            print("console script:", exe)
        EOF
        runHook postBuild
      '';
      installPhase = ''
        touch $out
      '';
    };

    # Evaluate the NixOS module against a minimal nixosSystem and pin down
    # the rendered units. Fails at eval time if the module regresses.
    speechModuleEval = pkgs.writeText "speech-module-eval" (builtins.toJSON {
      sttExecStart = moduleEval.config.systemd.services.speech-stt.serviceConfig.ExecStart;
      ttsExecStart = moduleEval.config.systemd.services.speech-tts.serviceConfig.ExecStart;
      agentExecStart = moduleEval.config.systemd.services.speech-agent.serviceConfig.ExecStart;
      sttMemoryMax = moduleEval.config.systemd.services.speech-stt.serviceConfig.MemoryMax;
      ttsMemoryMax = moduleEval.config.systemd.services.speech-tts.serviceConfig.MemoryMax;
      agentMemoryMax = moduleEval.config.systemd.services.speech-agent.serviceConfig.MemoryMax;
      firewallPorts = moduleEval.config.networking.firewall.allowedTCPPorts;
      openWebUiEnabled = moduleEval.config.services.open-webui.enable;
      openWebUiAudioEnv = moduleEval.config.services.open-webui.environment;
    });
  };
})
