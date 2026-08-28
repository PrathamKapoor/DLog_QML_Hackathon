# Hackathon Pitch — 3 minutes

**Rule for the delivery: the judge must understand the PROBLEM before hearing the word "quantum."** Do not open with quantum mechanics.

Have `python scripts/demo.py` already run in a visible terminal, and `figures/fig3_concept_geometry.png` open in another window.

---

## [0:00 – 0:30] The hidden coordinate

> "Here's a number: **h = g^x mod p**. Computing h from x is easy — it's modular exponentiation, your laptop does it instantly. Going *backwards*, recovering x from h, is the discrete logarithm problem. That's the assumption RSA-era cryptography is built on.
>
> Now I'm going to build a machine-learning problem out of that asymmetry. I label each point using a secret *interval of x*. So the answer depends **only** on x — but I hand the learner **only h**."

*(Point at the left panel of fig3.)*

> "This is what the model sees. Red and green, completely interleaved. No structure."

*(Point at the middle panel.)*

> "Same points. Same strip. I've just plotted them against the hidden coordinate x instead. One clean split. **The concept is a single interval** — a first-year student could learn it from ten examples.
>
> The problem was never hard. The *coordinate system* was hidden."

---

## [0:30 – 1:00] Classical models fail

> "So we threw eleven models at it. Tuned RBF-SVM with a full cross-validated grid search. Random forest, extra trees, gradient boosting. A neural net. Linear and polynomial kernels. k-NN.
>
> **Every single one lands on chance.** Best was 62% on 50 test points — and with 50 samples the standard error at chance is 7 points, so 62% is 1.7 sigma. It's noise. Not one of them clears p < 0.01."

---

## [1:00 – 1:30] The idea

> "Now the quantum feature map. Instead of encoding the bits of h, we build a superposition over a **window of the group's own orbit**: h, h·g, h·g², and so on.
>
> Building that uses only multiplication — the *easy* direction. **We never compute a discrete logarithm.** But because the state's support is a contiguous run along the exponent circle, the kernel between two points comes out as a **triangle in discrete-log distance.**
>
> The quantum state does the hard coordinate change for free. The SVM is then just solving interval membership."

*(Show fig2 — the kernel matrices.)*

> "Sort the training points by the hidden coordinate and look at the kernels. Ours: a clean diagonal band. The generic quantum kernel: the identity matrix. The classical RBF kernel: noise."

---

## [1:30 – 2:00] The result

> "Over 20 independent seeds — each one redrawing the secret *and* the data:
>
> **Coset kernel: 98.9%. Everything else: 49 to 54%. Separation: 38 points, worst seed 24.**
>
> That 38 is measured per seed against whichever model won *that* seed, which is the stricter way to count — comparing pooled means gives 45. The coset kernel is ahead on 20 out of 20.
>
> And these are paired tests on the same held-out items, not two accuracies eyeballed side by side. McNemar exact, p = 8.6 × 10⁻⁴."

---

## [2:00 – 2:40] We tried to break it — *the most important part*

> "But here's what I actually want to be judged on. We spent most of the effort trying to kill our own result.
>
> **One. Is it just 'quantum'?** We built a *strong* generic quantum kernel — entangling, 133 gates, depth 64, the Havlíček-style map everyone uses. Same pipeline, same everything. **It scores 50.3%.** Chance. So this is *not* 'quantum beats classical.'
>
> **Two. Is the trapdoor even real?** The obvious way to build this dataset is to label by the parity of the discrete log. **That version is broken** — the low bit of a discrete log is just the Legendre symbol, computable in log-p time by Euler's criterion. We ship the attack. It solves the parity version at **100%** and gets **46%** on ours. So we're not hiding behind 'it involves a discrete log, therefore it's hard.'
>
> **Three. Is the data just noise?** We trained a random forest on the hidden x directly. **99.2%.** So the labels are perfectly learnable *once you have the coordinate* — the failures are about the encoding, not about noise.
>
> **Four — and this is my favourite.** Run our quantum kernel on that broken parity concept. **It also drops to 46%.** Our kernel is *local* on the dlog circle; parity flips every step. **It's not a universal cheat.** It wins when its geometry matches the concept's geometry, and it fails when it doesn't."

---

## [2:40 – 3:00] What we do and don't claim

> "To be completely clear about scope: p is 1019. That's ten bits. It is **not** cryptographically hard — our own code runs baby-step giant-step on it as a data-integrity check. This is a classical simulation. No quantum hardware. And the state preparation we simulate is exponential; the efficient version is Shor's algorithm, which we cite but did not build.
>
> What we demonstrate is a measurable, reproducible learning separation on a finite instance, with the controls to back it up. The asymptotic claim is Liu, Arunachalam and Temme's theorem, and it's conditional on the discrete-log assumption. That's theirs, not ours.
>
> **The takeaway: quantum advantage is not automatic.** Our generic quantum kernel proves that in the same table as our headline. What mattered was matching the representation to the mathematics of the problem."

---

## Anticipated questions

**"Isn't 94% vs 62% just a lucky seed?"**
> That's why there are 20 seeds. 98.9% ± 1.8% versus 49–54%. Worst separation across all seeds: +24 points. And the paired McNemar test is on the same items, p = 8.6e-4.

**"Your baselines are weak."**
> Eleven of them, including a fully cross-validated RBF-SVM grid. Happy to add another live — the config is one YAML file. But the theory predicts *all* poly-time classical learners fail, so a stronger baseline would be a genuinely interesting result, not an embarrassment.

**"Isn't the quantum kernel cheating by using the discrete log?"**
> No — that's the key design point. The feature map is built by forward modular multiplication only. `CosetStateFeatureMap.statevector()` never calls `dlog()`. The closed-form *analysis* uses the oracle, and it's flagged as an auditing tool.

**"You're simulating classically, so is this really quantum?"**
> The kernel is exactly the fidelity kernel a quantum device would estimate; we verify that against literal compute–uncompute circuits to 1e-13. What's classically simulated is the *state preparation*, which is exponential in our implementation and `O(n³)` in Shor's. We're explicit about that in the README and LIMITATIONS.

**"Does this break cryptography?"**
> No, and we say so in the demo output. p = 1019 falls to brute force. This is a learning-theory demonstration.

**"Why does the coset kernel work at all?"**
> The overlap of two length-M arithmetic progressions on a circle of circumference N is `max(0, M−δ) + max(0, M−(N−δ))`. Square it and you have a triangular kernel in dlog distance. It's in `quantum_coset.py`, derived, and checked against a hand-counted set intersection in the tests.

**"How did you pick the window size?"**
> Fixed heuristic, M ≈ (p−1)/8, not tuned. The ablation sweeps the whole valid range: m = 2 through 8 all beat every classical baseline, and the default isn't even the best value.

---

## 60-second version

> "I built a classification problem that's trivially easy in one coordinate system and invisible in another — the two are related by a discrete logarithm. Eleven classical models get chance. A *generic* quantum kernel also gets chance. But a quantum feature map built from the group's own structure gets 98.9% over 20 seeds — a 38-point separation.
>
> Then we tried to break it: we broke a naive version of our own dataset with a number-theoretic attack, and we showed our quantum kernel fails on a concept that doesn't match its geometry. It's not a universal cheat.
>
> The lesson isn't 'quantum wins.' It's that the quantum representation has to match the mathematics — and we have the failed generic quantum kernel in the same table to prove it."
