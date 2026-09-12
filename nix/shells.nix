{
  nixpkgs,
  flake-utils,
  ...
}:
flake-utils.lib.eachDefaultSystem (system: let
  pkgs = import nixpkgs {
    inherit system;
    config.allowUnfree = true;
  };
in {
  # Plain dev shell: `uv` manages the venv from uv.lock. No CUDA is needed —
  # the speech stack (faster-whisper / kokoro-onnx / pipecat) is CPU-only here.
  devShells.default = pkgs.mkShell {
    buildInputs = with pkgs; [
      uv
    ];
  };
})
