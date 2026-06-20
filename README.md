# Hazard Compression: Catastrophic Forgetting in Diffusion-Based Generative Replay under Distribution Shift.

Accepted (Poster) at the ICML 2026 Workshop on Foundations of Deep Generative Models (FoGen).

---

## Summary

Diffusion models trained as generative replay buffers in reinforcement learning are vulnerable to a memorization failure we term **hazard compression**: as a Lagrangian safety penalty suppresses constraint-violating behavior, hazardous transitions vanish from the replay buffer, and the periodically retrained diffusion model catastrophically forgets the constrained region of state–action space.

We demonstrate this failure in Prioritized Generative Replay (PGR), introduce a diagnostic probe (**DiffHz**) that measures the diffusion model's retained hazard fidelity at the weight level, and show that a rare-event memory buffer resolves the feedback loop. On a second task where the Lagrangian multiplier diverges due to **integral windup** — a mechanistically distinct failure confirmed by DiffHz remaining high — combined λ-warmup and gradient clipping fully recovers 99.6% of unconstrained reward while reducing cost by 76%.

Together, DiffHz and the λ trajectory provide a lightweight diagnostic toolkit: **collapsing DiffHz signals generative forgetting; diverging λ signals control failure.**

---

## Contributions

1. **Hazard compression** — a policy-induced, selective forgetting of a safety-critical minority subpopulation in a continually retrained diffusion model, driven by distribution shift in the training data. We characterize its five-stage feedback loop and situate it alongside known memorization and continual-forgetting phenomena rather than as an entirely new effect.
2. **DiffHz** — a diagnostic probe that measures a diffusion model's retained fidelity to hazardous transitions at the *weight* level, preempting the counterargument that hazards are simply not being requested.
3. **Rare-event memory buffer** — resolves hazard compression by preserving hazardous transitions in the diffusion training set, reducing violations by 99.8% on Cheetah-run.
4. **Mechanistic distinction from integral windup** — on Walker-walk DiffHz correctly diagnoses a distinct failure mode (λ divergence), and combined λ-warmup + gradient clipping fully recovers unconstrained reward.

---

## Headline Results (33 runs, 3 seeds each, DeepMind Control Suite)

### Table 1 — Cheetah-run (hazard compression is the dominant failure)

Mean ± std across last 10 episodes, 3 seeds.

| Method            | Reward    | Ep. Cost   | DiffHz | λ   |
|-------------------|-----------|------------|--------|------|
| SAC               | 291 ± 20  | 0.0 ± 0.0  | N/A    | —    |
| PGR               | 679 ± 30  | 546 ± 70   | 13.3%  | —    |
| PGR+L             | 573 ± 6   | 4.1 ± 2.1  | 0.6%   | 3.08 |
| **PGR+L+Buffer**  | **561 ± 14** | **1.1 ± 0.4** | **8.6%** | **0.80** |

Rare buffer → 99.8% cost reduction, DiffHz preserved from collapse, λ 4× lower than Lagrangian-only.

### Table 2 — Walker-walk (integral windup + anti-windup ablation)

| Method            | Reward      | Ep. Cost    | DiffHz | λ      |
|-------------------|-------------|-------------|--------|--------|
| SAC               | 855 ± 126   | 178 ± 184   | N/A    | —      |
| PGR               | 933 ± 37    | 109 ± 102   | 88.0%  | —      |
| PGR+L             | 179 ± 26    | 27 ± 3      | 73.7%  | 246.8  |
| PGR+L+Buffer      | 194 ± 7     | 30 ± 9      | 74.2%  | 224.4  |
| +Warmup           | 219 ± 17    | 25 ± 8      | 82.1%  | 79.5   |
| +Clip             | 808 ± 118   | 23 ± 5      | 85.0%  | 8.7    |
| **+WarmClip**     | **929 ± 20** | **26 ± 6**  | **91.2%** | **6.2** |

+WarmClip recovers 99.6% of unconstrained PGR reward (Welch's t-test p = 0.91, i.e. statistically indistinguishable from the unconstrained baseline) while keeping λ at 6.2 versus the uncontrolled 224.

---

## Repository Structure

```
pgr/                                  Code base (extends PGR, ICLR 2025)
  safety/
    online_cost_cond.py               Main training loop (SAC + diffusion retrain + cost)
    cost_agent.py                     REDQ-SAC agent with Lagrangian + anti-windup
    cost_replay_buffer.py             CostReplayBuffer + RareEventBuffer
    hazard_wrapper.py                 DMC env wrappers with binary cost signals
    run_parallel.py                   Multi-worker experiment launcher
    make_figures.py                   Paper figure generation
  synther/diffusion/
    elucidated_diffusion.py           EDM diffusion + trainer
    denoiser_network_cond.py          Conditional denoiser
  config/online/                      Gin configs (sac.gin, sac_cond_synther_dmc.gin)
  requirements.txt                    Pinned Python dependencies

results/cheetah-run-v0/               12 Cheetah result JSONs (4 methods × 3 seeds)
results_walker/walker-walk-v0/        21 Walker result JSONs (7 methods × 3 seeds)

submit.pdf                            Workshop submission PDF
```

---

## Installation

```bash
pip install -r pgr/requirements.txt
```

Tested on Python 3.12 with PyTorch 2.x (CUDA or CPU/MPS).

---

## Reproducing the Main Results

### Single run — the best-performing method (+WarmClip on Walker-walk)

```bash
cd pgr
python safety/online_cost_cond.py \
  --env walker-walk-v0 --seed 42 \
  --gin_config_files config/online/sac_cond_synther_dmc.gin \
  --gin_params 'redq_sac.cond_top_frac = 0.25' \
  --use_rare_buffer --use_lagrangian \
  --lambda_warmup_episodes 20 --lambda_grad_clip 0.1 \
  --method_name pgr_lb_warmclip \
  --results_folder ../results_walker/walker-walk-v0
```

Drop `--lambda_warmup_episodes` and/or `--lambda_grad_clip` to run the ablation variants (`pgr_lb`, `+Warmup`, `+Clip`).

### Full campaign (33 runs)

```bash
cd pgr

# Cheetah — 12 runs
python safety/run_parallel.py \
  --envs cheetah-run-v0 \
  --methods sac pgr pgr_lagrangian pgr_lb \
  --seeds 42 123 456 --epochs 100 \
  --results_dir ../results

# Walker — 21 runs
python safety/run_parallel.py \
  --envs walker-walk-v0 \
  --methods sac pgr pgr_lagrangian pgr_lb pgr_lb_warmup pgr_lb_clip pgr_lb_warmclip \
  --seeds 42 123 456 --epochs 100 \
  --results_dir ../results_walker
```

---

## Key Flags

| Flag | Description |
|---|---|
| `--use_rare_buffer` | Enable 500-slot FIFO buffer of hazardous transitions (20% mix into diffusion retrain batches) |
| `--use_lagrangian` | Enable dual gradient ascent on λ (η = 0.01, cost limit d = 2.0) |
| `--lambda_warmup_episodes N` | Hold λ = 0 for the first N episodes (anti-windup) |
| `--lambda_grad_clip X` | Cap \|Δλ\| ≤ X per episode (anti-windup) |
| `--method_name NAME` | Override result filename prefix |

---

## Results Data

All 33 result JSONs (100 episodes × reward / cost / λ / DiffHz per run) are tracked in the repo under `results/` and `results_walker/`:

```python
import json
d = json.load(open('results_walker/walker-walk-v0/pgr_lb_warmclip_seed42_results.json'))
d.keys()
# episode_rewards, episode_costs, lambda_log, diffhz_log, method, seed, ...
```

---

## Built On

- Prioritized Generative Replay (ICLR 2025)
- SynthER / REDQ
- Elucidated Diffusion Models (Karras et al., 2022)
- PID Lagrangian methods for safe RL (Stooke et al., ICML 2020)
