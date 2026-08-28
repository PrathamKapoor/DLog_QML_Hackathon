# Formal ablation table

| component | variant | accuracy | delta_vs_full | question | interpretation |
|---|---|---|---|---|---|
| Full system | coset kernel, default window | 0.9400 | 0.0000 | reference configuration | structure-aware representation |
| Representation | generic Rz/Rx/CZ feature map | 0.6000 | -0.3400 | Would any expressive quantum feature map do? | geometry mismatch -- being quantum is not sufficient |
| Representation | best classical (Linear-SVM) | 0.6200 | -0.3200 | Is a classical kernel on the same bits enough? | no access to the dlog coordinate |
| Coset window | m=1 (M=2) | 0.7400 | -0.2000 | Does the result hinge on a tuned window size? | too narrow: near-orthogonal kernel |
| Coset window | m=2 (M=4) | 0.8800 | -0.0600 | Does the result hinge on a tuned window size? | robust |
| Coset window | m=3 (M=8) | 0.9800 | 0.0400 | Does the result hinge on a tuned window size? | robust |
| Coset window | m=4 (M=16) | 0.9600 | 0.0200 | Does the result hinge on a tuned window size? | robust |
| Coset window | m=5 (M=32) | 0.9400 | 0.0000 | Does the result hinge on a tuned window size? | robust |
| Coset window | m=6 (M=64) | 0.9400 | 0.0000 | Does the result hinge on a tuned window size? | robust |
| Coset window | m=7 (M=128) | 0.9400 | 0.0000 | Does the result hinge on a tuned window size? | default |
| Coset window | m=8 (M=256) | 0.9400 | 0.0000 | Does the result hinge on a tuned window size? | robust |
| Target concept | coset kernel on parity concept | 0.4600 | -0.4800 | Is the coset kernel a universal cheat? | local kernel vs alternating concept -- fails |
| Target concept | Legendre attack on parity concept | 1.0000 | 0.0600 | Is every dlog-derived label actually hard? | parity has no trapdoor -- broken in O(log p) |
