# cifar10-fast

This is a fork for the purposes of profiling.

## How to reproduce

1. Run `nix-shell` to get the correct environment.
2. Run `nixGLNvidia-510.47.03 python dawn.py` to run the training/profiling script.

When running on non-NixOS systems you'll likely need nixGL installed as well. On NixOS you should be able to get away without it.

All results are run on a p3.2xlarge EC2 instance. For convenience the results are already checked in to the `profiling/` subdirectory.

## Issues

I get the following error when attempting to create a flamegraph with the stack traces saved by PyTorch:

```
[nix-shell:~/dev/cifar10-fast]$ flamegraph.pl --title "pytorch single epoch" --countname "us." profiling/stacks_5.txt
ERROR: No stack counts found
<?xml version="1.0" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg version="1.1" width="1200" height="60" onload="init(evt)" viewBox="0 0 1200 60" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<!-- Flame graph stack visualization. See https://github.com/brendangregg/FlameGraph for latest version, and http://www.brendangregg.com/flamegraphs.html for examples. -->
<!-- NOTES:  -->
<text  x="600.00" y="24" >ERROR: No valid input provided to flamegraph.pl.</text>
</svg>
```
