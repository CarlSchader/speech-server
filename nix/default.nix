{flake-utils, ...} @ inputs:
flake-utils.lib.meld inputs [
  ./env.nix
  ./shells.nix
]
