# Metrics

Every measured number for the published run (training 2026-08-25, full-test
autopsy 2026-08-29). Each section names its source; regeneration commands are
at the bottom.

## Headline results

| Metric | Value | Source |
|---|---|---|
| Single-step exact match, 500-sample eval | 98.0% | `eval.py` |
| Single-step exact match, full 3,000-row test | **99.63%** (2,989/3,000) | `autopsy.py` |
| "State unchanged" trivial baseline | 42.6% | `eval.py` |
| Rollout exact match at step 10 (sampled, 50 eps) | 96.0% | `eval.py` → `eval_drift.png` |
| Rollouts clean through all 10 steps (full, 150 eps) | **148/150** | `autopsy.py` |

Grading is byte-equality of the full predicted state string; greedy decoding.

## Per-rule accuracy (full test set)

| Rule | Train examples | Test | Misses | Accuracy |
|---|---:|---:|---:|---:|
| move: succeeds | 20,487 | 1,759 | 0 | 100% |
| move: blocked by edge | 2,405 | 217 | 0 | 100% |
| **move: blocked by wall** | **1,230** | **112** | **11** | **90.18%** |
| move: blocked by locked door | 1,089 | 89 | 0 | 100% |
| pick up: succeeds | 1,314 | 116 | 0 | 100% |
| pick up: no-op | 3,777 | 293 | 0 | 100% |
| unlock: succeeds | 813 | 73 | 0 | 100% |
| unlock: no-op | 3,885 | 341 | 0 | 100% |

Source: `breakdown.py` → `breakdown.md`, `breakdown.png`.

## Miss forensics

- 13 misses total (11 single-step + 2 rollout), all the same kind:
  **phantom change** — the transition was a wall-block no-op, the model moved
  the agent anyway. Wrong field: `agent`, 13/13.
- By action: move south 10, move east 2, move west 1.
- Only 3 unique (state, action) pairs; episode 1769 repeats its miss 10×.
- Wall-blocks by direction in training: west 359, north 344, east 301,
  **south 226** — the rarest direction is the most-missed one.

Source: `autopsy_report.txt`, `autopsy_misses.jsonl` (raw, unfiltered).

## Dataset

| | train | test |
|---|---:|---:|
| Transitions | 35,000 | 3,000 |
| Episodes (20 steps each, split by episode) | 1,750 | 150 |
| No-op transitions | 12,386 (35.4%) | 1,052 (35.1%) |
| States with door locked | 29,571 (84.5%) | 2,491 (83.0%) |
| Episode outcomes: door opened / key only / neither | 813 / 501 / 436 | 73 / 43 / 34 |
| Mean step of first pick-up / first unlock | 8.61 / 12.32 | 8.52 / 12.03 |

Action distribution (train): move east 6,312 · north 6,359 · south 6,187 ·
west 6,353 · pick up 5,091 · unlock 4,698. No-op rates are lopsided by
design of the world, not the sampler: unlock is a no-op 3,885/4,698 (~83%),
pick up 3,777/5,091 (~74%), moves ~19%.

Source: computed from `data/train.jsonl` / `data/test.jsonl` (seed 42,
regeneration is byte-identical).

## Model & adapter

| | |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B` (494,032,768 weights, frozen) |
| LoRA | r=16 on q/k/v/o projections, all 24 layers |
| Adapter tensors | 192 (one A + one B per projection per layer) |
| Trainable parameters | 2,162,688 (0.44% of total) |
| Adapter file size | 8,676,008 bytes |

Source: `checkpoints/qwen05b-lora/adapter_model.safetensors` header (readable
with 6 lines of stdlib Python — see the safetensors snippet in the blog post).

## Training run

lr 2e-4 · 3 epochs · batch 64 · 1,641 optimizer steps · completion-only loss
(prompt masked with −100) · final train loss ≈ 0.003 · ~190 examples/sec on a
rented RTX 4090 · ~10 min wall-clock · ≈ $0.15 pod cost.

Source: recorded from the training run log; reproduce with `train.py` or
`podenv/pod.py run`.

## Regenerate

```bash
python breakdown.py    # per-rule table + chart + exhibits (no model call)
python eval.py         # sampled eval + drift curve (CPU/MPS ok)
python autopsy.py      # full-test re-grade (GPU recommended)
python gen_data.py     # regenerate datasets; diff against data/ to verify
python train.py        # retrain from scratch (~10 min on a 4090)
```
