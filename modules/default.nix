{...}: {
  nixosModules = rec {
    speech = import ./speech.nix;
    default = speech;
    # Opinionated preset for the DGX Spark host (import alongside `speech`).
    dgx-spark = import ./dgx-spark.nix;
  };
}
