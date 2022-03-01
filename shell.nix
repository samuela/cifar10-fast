# Run with nixGL, eg `nixGLNvidia-510.47.03 python cifar10_convnet_run.py --test`

let
  # Last updated: 1/27/2022. Check for new commits at status.nixos.org.
  # pkgs = import (/home/skainswo/dev/nixpkgs) { };
  # Changes on top of nixpkgs upstream:
  # - merged in https://github.com/NixOS/nixpkgs/pull/162277
  pkgs = import (fetchTarball "https://github.com/samuela/nixpkgs/archive/23fdac1f4d4dbaa3c47446550952eb2298199eb4.tar.gz") {
    config.allowUnfree = true;
    overlays = [
      (final: prev: {
        cudatoolkit = prev.cudatoolkit_11_5;
        cudnn = prev.cudnn_8_3_cudatoolkit_11_5;
      })
    ];
  };
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    flamegraph
    python3
    python3Packages.ipython
    python3Packages.jupyter
    python3Packages.pandas
    python3Packages.pytorch-bin
    python3Packages.torchvision
    yapf
  ];

  # See
  #  * https://discourse.nixos.org/t/using-cuda-enabled-packages-on-non-nixos-systems/17788
  #  * https://discourse.nixos.org/t/cuda-from-nixkgs-in-non-nixos-case/7100
  #  * https://github.com/guibou/nixGL/issues/50
  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.cudatoolkit_11_5}/lib
  '';
}
