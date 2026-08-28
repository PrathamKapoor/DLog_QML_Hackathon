# Reference run: held-out accuracy

| model | accuracy | wilson_ci_95 | binomial_p | above_chance_at_1pct | role |
|---|---|---|---|---|---|
| Hidden-x oracle (control) | 0.9600 | [0.865, 0.989] | 0.0000 | True | oracle control (not a baseline) |
| Quantum coset kernel | 0.9400 | [0.838, 0.979] | 0.0000 | True | structure-aligned quantum |
| Linear-SVM | 0.6200 | [0.482, 0.741] | 0.1189 | False | classical baseline |
| RBF-SVM (tuned) | 0.6000 | [0.462, 0.724] | 0.2026 | False | classical baseline |
| Quantum generic kernel | 0.6000 | [0.462, 0.724] | 0.2026 | False | generic quantum control |
| Poly-SVM (d=3) | 0.5800 | [0.442, 0.706] | 0.3222 | False | classical baseline |
| MLP | 0.5200 | [0.385, 0.652] | 0.8877 | False | classical baseline |
| RBF-SVM | 0.5000 | [0.366, 0.634] | 1.0000 | False | classical baseline |
| ExtraTrees | 0.5000 | [0.366, 0.634] | 1.0000 | False | classical baseline |
| kNN (k=5) | 0.4800 | [0.348, 0.615] | 0.8877 | False | classical baseline |
| GradientBoosting | 0.4600 | [0.330, 0.596] | 0.6718 | False | classical baseline |
| Legendre attack (control) | 0.4600 | [0.330, 0.596] | 0.6718 | False | control |
| RandomForest | 0.4200 | [0.294, 0.558] | 0.3222 | False | classical baseline |
