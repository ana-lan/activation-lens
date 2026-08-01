# ActivationLens

**Mechanistic interpretability meets inference optimization on GPT-2-small.**

![Python](https://img.shields.io/badge/python-3.11-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Status](https://img.shields.io/badge/status-research%20prototype-orange)

[Live Demo](#) · [Results](#results-at-a-glance) · [Result Artifacts](#result-artifacts)

---

## Results at a glance

| Experiment | Result |
|---|---|
| Simulated 16-bit rounding | 99.83% recall@5; 1.000 mean feature correlation; behavior preserved |
| Simulated 8-bit rounding | 29.44% recall@5; 0.641 correlation; perplexity increased 1.99× |
| Simulated 4-bit rounding | 0.01% recall@5; 0.004 correlation; model behavior collapsed |
| Best single harm-associated feature | AUROC 0.653 |
| 100-feature regularized linear probe | AUROC 0.796; AUPRC 0.801 on untouched holdout |
| Live-prefix probe | AUROC 0.758; recall 91.6%; FPR 65.2% |
| Optimized monitoring kernel | +0.66% mean overhead (CPU benchmark); 95% CI −0.29% to 1.61% |
| SAE decoder-direction steering | No target-specific effect distinguishable from matched random perturbation |

---

## What this project asks

1. Does simulated weight quantization degrade model behavior and SAE feature structure together?
2. Can SAE features support an online, low-overhead safety probe?
3. Does increasing an interpretable SAE coordinate causally steer generated behavior?

Built end-to-end on GPT-2-small: activation extraction (TransformerLens), a sparse autoencoder (SAE) trained from scratch, a FastAPI + Next.js interactive dashboard, and three rigorously benchmarked experiments.

---

## Foundations — building the interpretability pipeline

Before any of the three benchmark experiments, this project builds the core infrastructure from scratch:

**Activation extraction.** Hooks into GPT-2-small's layer-6 residual stream via TransformerLens, capturing ~1.74M activation vectors from OpenWebText for SAE training.

**SAE training.** A sparse autoencoder (768 → 8,192 features) trained from scratch with an encoder/decoder architecture and an L1 sparsity penalty. Early training runs showed 52–82% of features going permanently dead. After adding decoder weight normalization and periodic dead-feature resampling, dead-feature prevalence fell to 0.77% while reconstruction quality remained high. (No controlled ablation isolated the individual contribution of each fix — both were added together in response to the diagnosed problem.)

**Interactive dashboard.** A FastAPI backend + Next.js/TypeScript/Tailwind frontend for browsing individual SAE features and their top-activating examples, built on the same hook/SAE pipeline used throughout every later experiment.

This groundwork is what makes the three benchmark phases below possible. It also exposed an engineering constraint: TransformerLens warned that the installed PyTorch/MPS combination could produce incorrect results, so all final reported experiments were run on CPU (see [Environment & hardware](#environment--hardware)).

---

## Phase 5 — Does quantization break interpretability?

**Method:** Trained an SAE on GPT-2-small's layer-6 residual stream. Simulated 16-bit/8-bit/4-bit quantization via uniform min-max weight rounding. Production quantization backends such as GPTQ/AWQ were not evaluated; this phase isolates uniform min-max weight rounding specifically (see [Limitations](#limitations)). Evaluated on 500 held-out documents (zero overlap with SAE training data), using three independent metrics: top-k activating-position recall, per-feature Pearson correlation, and behavioral checks (perplexity, KL divergence, token agreement).

| Precision | Perplexity ratio | Top-1 agreement | Mean feature correlation | Recall@5 |
|---|---|---|---|---|
| 16-bit | 1.00× | 99.92% | 1.000 | 99.83% |
| 8-bit | 1.99× | 59.33% | 0.641 | 29.44% |
| 4-bit | 32,328× | 0.38% | 0.004 | 0.01% |

**Finding:** Under naive weight rounding, behavioral and representational quality degrade *together*, not independently. The 16-bit result is a clean negative control validating the whole pipeline. 4-bit rounding destroys the model outright — informative about this quantization method's limits, not (on its own) evidence of a hidden interpretability danger.

---

## Phase 6 — A live, low-overhead safety monitor

**6.1–6.2 — Single-feature discovery.** Searched all 8,192 SAE features against BeaverTails (harmful vs. safe response text), using a strict discovery/validation/test split (no data leakage). The threshold was selected on validation to maximize F1. On the frozen test set, this threshold produced 92.0% recall and an 84.4% false-positive rate — a real, weak, confounded signal, not a working monitor (best feature AUROC: 0.653).

**100-feature regularized linear probe.** A logistic-regression probe over the top 100 finite features by absolute Cohen's d on the discovery split, evaluated once on a genuinely untouched holdout (96 of 100 coefficients nonzero — a regularized probe, not a sparse one):

| Model | AUROC | AUPRC | Recall | FPR |
|---|---|---|---|---|
| Single feature (#3149) | 0.653 | 0.652 | 92.0% | 84.4% |
| 100-feature probe | **0.796** | **0.801** | 91.2% | 60.8% |

**6.3 — Live conversion.** Converted the probe into an online, per-token monitor (running-mean feature tracking, prefix-based calibration on a separate holdout). This is online per-token monitoring with a post-generation trace UI — the browser receives the full trace after generation completes, not a streamed live view. Frozen live-prefix AUROC: **0.758** (recall 91.6%, FPR 65.2%).

**6.5 — Overhead benchmark.** Ten paired, counterbalanced trials, greedy decoding, CPU inference (4 PyTorch threads):

| Condition | Throughput | Overhead vs. baseline |
|---|---|---|
| Baseline (no monitoring) | 12.94 tok/s | — |
| Naive full-SAE monitor | 12.44 tok/s | +4.12% (95% CI: 1.44–6.80%) |
| **Optimized (100-feature only)** | 12.86 tok/s | **+0.66% (95% CI: −0.29% to 1.61%)** |

**Finding:** an optimized one-pass monitoring kernel — computing only the 100 needed encoder features, skipping the full SAE decoder and 8,092 unused features — added +0.66% mean overhead in this CPU benchmark, eliminating ~84% of the naive implementation's overhead. This result describes the optimized benchmark kernel specifically; the current `/safety/monitor` API endpoint uses a separate, correctness-first two-pass implementation and has not yet adopted this optimization.

**Limitations specific to this phase:**
- BeaverTails evaluation uses response text from a single dataset
- The live monitor is calibrated on standalone responses but applied to prompt-conditioned generation — a real, uncorrected distribution shift
- No raw-residual or plain-text-classifier baseline was evaluated, so these results do not establish that SAE features are uniquely useful versus simpler alternatives
- High recall comes with an operationally unacceptable 65.2% false-positive rate — this is a research prototype, not a deployable monitor

---

## Phase 7 — Feature steering

Inspired by Anthropic's Golden Gate Claude experiment, but not an exact replication. Anthropic directly clamped learned feature activations (e.g., the Golden Gate Bridge feature, to a high fixed value) and observed thematic behavior ([Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/), [Golden Gate Claude announcement](https://www.anthropic.com/news/golden-gate-claude)). ActivationLens instead evaluates normalized additive SAE decoder-direction interventions at GPT-2-small's layer-6 residual stream — a related but not equivalent intervention.

**Method:** Five semantically screened features were evaluated under last-position steering across 4 strength levels and 5 neutral prompts. Three finalists — military, books, and the lexical feature " Mr." — were then tested under persistent all-position intervention, alongside one matched random unit direction.

**Finding:** Last-position tests verified that intervention strength reliably increased the targeted encoded coordinate. However, neither intervention policy produced consistent target-concept intrusion. Persistent target directions caused generic output changes and high-strength degeneration qualitatively comparable to the random control.

**Interpretation:** within this model, SAE, layer, candidate set, prompt suite, and decoding policy, internal coordinate control did not translate into reliable concept-level control of generation. Two untested hypotheses for why: (1) GPT-2-small's SAE may lack sufficiently clean, monosemantic directions at this scale; (2) layer 6 may not be the right intervention point for steering specifically, even if reasonable for reading features.

---

## Architecture

```
Base LLM (GPT-2-small, frozen)
         ↓
  Hook/Capture layer ──────────────→ Quantized variants (16/8/4-bit)
         ↓                                    ↓
    SAE (trained once) ←──────────────────────┘
         ↓
  ┌──────┴──────┬─────────────┬─────────────┐
  Feature        Safety         Steering       Benchmark
  Dashboard      Monitor        (Phase 7)      Layer
  (FastAPI +     (100-feature                  (feature survival,
   Next.js)      live probe)                    overhead, dose-response)
```

## Tech stack

Python · PyTorch · TransformerLens · scikit-learn · FastAPI · Next.js · TypeScript · Tailwind CSS

## Environment & hardware

- Apple Silicon Mac; CPU-only experimental runs
- TransformerLens warned that the installed PyTorch/MPS combination could produce incorrect results, so all final reported experiments were run on CPU
- Python 3.11 with a conda environment
- Overhead benchmark: 4 PyTorch CPU threads, no KV cache
- Thread counts for other experiments were not explicitly standardized

## Limitations

- Quantization is simulated via uniform min-max weight rounding. Production quantization backends such as GPTQ/AWQ were not evaluated.
- Single model (GPT-2-small), single layer (6), single SAE training seed
- Steering tested at one layer only; negative result may not generalize to other layers/models
- Behavioral evaluation uses greedy decoding only; no KV-caching in any benchmark
- Safety monitor has no baseline comparison against simpler methods (raw residual, plain-text classifier)

## Implemented vs. Future work

**Implemented:** activation extraction and SAE training pipeline with dead-feature diagnosis and resolution (resampling + decoder normalization); simulated quantization benchmark (16/8/4-bit) with held-out behavioral and feature-level evaluation; single- and multi-feature harm classifiers with strict train/validation/test discipline; live per-token safety monitor with calibrated overhead benchmark; SAE feature-steering experiment with random-direction control; interactive feature-browser and safety-monitor dashboard.

**Future work:**
- Real quantization backends (GPTQ/AWQ) to test whether behavior-preserving quantization still degrades features
- Steering on a larger open model (Llama 3 8B / Gemma 2) with a correspondingly larger SAE
- Layer sweep for both interpretability and steering
- Multi-seed SAE training, category-specific harm analysis, causal feature-transfer under quantization
- Raw-residual and plain-text-classifier baselines for the safety monitor
- KV-caching and batching benchmarks; speculative decoding + steering interaction
- Adopt the optimized one-pass monitoring kernel in the live `/safety/monitor` API endpoint

## Result artifacts

| Experiment | Script | Committed summary | Local artifact |
|---|---|---|---|
| SAE training | `model/train_sae.py` | — | `model/sae_weights.pt` (gitignored; regenerate via script) |
| Feature survival (quantization) | `benchmarks/feature_survival.py` | `results/quantization_summary.json` | `results/*_metrics.pt` |
| Harm-feature discovery | `safety/find_harm_feature.py` | `results/safety_probe_summary.json` | `results/harm_feature_discovery.pt` |
| Multi-feature classifier | `safety/multi_feature_classifier.py` | `results/safety_probe_summary.json` | `results/multi_feature_classifier.pt` |
| Live monitor calibration | `safety/calibrate_live_multifeature.py` | `results/live_monitor_summary.json` | `results/live_multifeature_calibration.pt` |
| Overhead benchmark | `safety/benchmark_overhead.py` | `results/overhead_summary.json` | `results/overhead_benchmark.pt` |
| Steering — last position | `steering/stage2_test_candidates.py` | `results/steering_last_position.json` | — |
| Steering — persistent/random control | `steering/stage2_persistent_test.py` | `results/steering_persistent.json` | — |
| Steering candidate screen | `steering/screen_candidates.py` | — | `results/steering_candidate_screen.pt` |

Each committed JSON summary includes: configuration and random seed, dataset slices/splits used, model/layer/SAE dimensions, the reported metrics, relevant feature IDs and strengths, prompt/completion pairs (for steering), random-control outputs (for steering), and phase-specific limitations.

## Reproducing this

```bash
git clone https://github.com/ana-lan/activation-lens
cd activation-lens
conda create -n activation-lens python=3.11 -y
conda activate activation-lens
pip install -r requirements.txt

# Build the training activation cache and train the SAE
python3 -m model.build_activation_cache
python3 -m model.train_sae

# Phase 5 — quantization benchmark
python3 -m benchmarks.feature_survival

# Phase 6 — safety monitor (depends on artifacts from prior commands; can take substantial time on CPU)
python3 -m safety.find_harm_feature
python3 -m safety.multi_feature_classifier
python3 -m safety.calibrate_live_multifeature
python3 -m safety.benchmark_overhead

# Phase 7 — steering
python3 -m steering.screen_candidates
python3 -m steering.stage2_test_candidates
python3 -m steering.stage2_persistent_test

# Dashboard
uvicorn backend.api.main:app --reload
# in a second terminal:
cd frontend
npm install
npm run dev
```

## License

MIT — see [LICENSE](./LICENSE).