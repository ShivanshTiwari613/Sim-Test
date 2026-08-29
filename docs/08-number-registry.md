# 08 · The number registry — every constant, decoded

Numbers in this project belong to **five species** that look identical on
screen. When a number confuses you, first ask which species it is:

| species | examples | nature |
|---|---|---|
| **world sizes** | 5, 20, 35,000 | real quantities — counting things that exist |
| **name tags** | episode 720 | identifiers — labels, never math (720 isn't "bigger" than 7) |
| **dictionary indices** | 16061, 18, 19 | token ids — row numbers in Qwen's vocabulary; size meaningless |
| **sentinels** | −100, 151643 | magic values with agreed meanings |
| **dials** | 16, 64, 2e-4, 3 | hyperparameters — human choices that could be otherwise |

## The registry

### World (env.py)
| number | what it really is |
|---|---|
| 5 | grid width/height → 25 cells |
| 6 | actions: 4 moves + pick up + unlock |
| 0–2 | walls per episode, `randint(0, 2)` |
| 1 | Manhattan distance required to unlock |

### Data generation (gen_data.py)
| number | what it really is |
|---|---|
| 42 | RNG seed (dial) — different seed → different but equally valid dataset |
| 20 | steps per episode |
| 1,750 / 150 | train / test episode counts; split boundary is `ep < 1750` |
| 0.6 | reflex coins: P(pick up on key), P(unlock beside door) |
| 0.35 | compass coin: P(step toward current goal) |
| 35,000 / 3,000 | rows: 1,750×20 and 150×20 (counting, not id math) |
| 0–1,899 | episode ids (name tags) — never shown to the model |
| 35.4% / 12,386 | no-op rows — a third of the physics is "nothing happens" |

### Tokenizer (Qwen)
| number | what it really is |
|---|---|
| 151,936 | dictionary size (vocabulary) |
| 18 / 19 | the tokens "3" and "4" — each digit is its own row |
| 151643 | `<|endoftext|>` — the trained stop sign (sentinel) |
| 45 + 36 = 81 | prompt + completion tokens of the example row |

### Training tensors (train.py)
| number | what it really is |
|---|---|
| −100 | "don't grade this position" (sentinel) — masks all prompt tokens |
| 512 | max_len safety cap (dial); real rows are ~81–120 so it never bites |
| 64 | batch size (dial); was 8 → starved the 4090 |
| 1 / 0 | attention mask: look / ignore (0 only over padding) |

### Model & LoRA
| number | what it really is |
|---|---|
| 896 | Qwen hidden size — width of the signal between layers |
| 24 | transformer layers, each getting 4 LoRA detours |
| 16 | LoRA rank r (dial) — width of the trainable bottleneck |
| 32 | lora_alpha = 2×r (convention dial) |
| 0.05 | LoRA dropout (dial) |
| 2,162,688 | trainable params = (28,672+16,384+16,384+28,672) × 24 layers |
| 496,195,456 | total params with LoRA attached → 0.44% trains |
| 8.7 MB | the saved adapter (a patch, not a model) |

### Schedule & results
| number | what it really is |
|---|---|
| 3 | epochs (dial) |
| 2e-4 | learning rate = 0.0002 (dial) — fine for LoRA-only updates |
| 1,641 | optimizer steps: ceil(35,000/64) × 3 |
| ~0.003 → 0.0 | final train loss — correct for a deterministic, canonical world |
| 98.0% / 42.6% | model vs "unchanged" baseline, single-step exact match |
| 96% / 48-of-50 | rollout accuracy at step 10 / episodes perfect throughout |
| $0.15 / 10 min | run cost / training wall-clock on the 4090 |

*Interactive version with the clickable 81-token inspector:*
https://claude.ai/code/artifact/eee36dc4-9700-4364-a660-70b244f1bae0
