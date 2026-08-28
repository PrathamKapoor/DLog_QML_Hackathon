# Archived: single-file prototype (superseded)

This directory holds the first version of the project — one 1,300-line script
(`generate_and_evaluate.py`) plus its own README, requirements and outputs.

**It is not the submission.** The authoritative artifact is the parent project:
`DLog_QML_Hackathon.ipynb` backed by `src/dlog_qml/`.

It is kept only because it is the record of where the two mathematical
corrections that shape the current design were first found:

1. **dlog parity is not a hard label.** The original brief specified
   `y = (-1)^(log_g h mod 2)`. The least significant bit of a discrete log in
   `Z_p*` is the quadratic-residuosity bit, recoverable in `O(log p)` by
   Euler's criterion, so that target has no trapdoor. It became the
   `--labeling parity` control in the current code.

2. **An `Rz/Rx/CZ` ladder feature map cannot solve this task.** The brief
   specified one as *the* quantum kernel. It scores at chance, and now serves
   as the generic negative control that makes the headline falsifiable.

The prototype's numbers are consistent with the current code (coset 0.940,
generic 0.600, best classical 0.620 on the reference seed) but it lacks the
20-seed sweep, the leakage guard, the kernel-geometry diagnostics, the ablation
table and the 182-test suite. Use the parent project.
