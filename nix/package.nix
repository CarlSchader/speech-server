# Wheel environment shared by flake outputs and downstream packagers.
# The x86_64 Numba wheel needs TBB; do not ignore missing ELF deps.
{
  pkgs,
  uv2nix,
  pyproject-nix,
  pyproject-build-systems,
  cuda ? false,
}: let
  workspace = uv2nix.lib.workspace.loadWorkspace {workspaceRoot = ../.;};
  pythonSet =
    (pkgs.callPackage pyproject-nix.build.packages {python = pkgs.python312;}).overrideScope
    (pkgs.lib.composeManyExtensions [
      pyproject-build-systems.overlays.default
      (workspace.mkPyprojectOverlay {sourcePreference = "wheel";})
      (_final: prev: {
        # Pipecat 1.10's /start ignores --proxy for WebSocket URLs and
        # advertises loopback:port to browsers behind HTTPS. Fail on drift.
        pipecat-ai = prev.pipecat-ai.overrideAttrs (old: {
          postInstall =
            (old.postInstall or "")
            + ''
              substituteInPlace "$out/${pkgs.python312.sitePackages}/pipecat/runner/run.py" \
                --replace-fail 'wsUrl=f"{scheme}://{args.host}:{args.port}/ws-client",' \
                  'wsUrl=f"wss://{args.proxy}/ws-client" if args.proxy else f"{scheme}://{args.host}:{args.port}/ws-client",'
            '';
        });
        numba = prev.numba.overrideAttrs (old: {
          buildInputs = (old.buildInputs or []) ++ [pkgs.tbb];
        });
        onnxruntime =
          if cuda
          then
            prev.onnxruntime.overrideAttrs (old: {
              src = pkgs.fetchurl {
                url = "https://files.pythonhosted.org/packages/d0/2c/5b3fd4748cf7ed291eae541a37e426efc20ea04cb6e6a05768304ab0aa41/onnxruntime_gpu-1.24.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
                hash = "sha256-6w448MHvO3auAIHI5R7tIN2JJaqRbw/G+bixfQVhDpk=";
              };
              buildInputs = (old.buildInputs or []) ++ (with pkgs.cudaPackages_12; [cuda_cudart libcublas libcufft libcurand cudnn]);
              # We support CUDA EP, not TensorRT. Remove that optional plugin
              # instead of suppressing its unresolved native dependencies.
              postInstall =
                (old.postInstall or "")
                + ''
                  rm "$out/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime_providers_tensorrt.so"
                '';
            })
          else prev.onnxruntime;
      })
    ]);
in
  assert !cuda || pkgs.stdenv.hostPlatform.system == "x86_64-linux";
    pythonSet.mkVirtualEnv "speech-env" workspace.deps.all
