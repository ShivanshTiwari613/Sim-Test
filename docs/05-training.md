# 05 · train.py — teaching Qwen the rulebook

Strip the plumbing and train.py contains two ideas: **grade only the answer**
(label masking, see doc 04) and **train 0.44% of the model** (LoRA). The rest
is scaffolding.

## The cast

- **Base model**: Qwen2.5-0.5B — ~494M frozen parameters, hidden size 896,
  24 transformer layers, vocab 151,936.
- **Dataset**: `TransitionDataset` wraps the 35,000 rows; each `__getitem__`
  builds `STATE: …\nACTION: …\nNEXT: ` + completion, tokenizes the two parts
  separately (so it knows exactly where to place the -100 mask boundary), and
  appends `<|endoftext|>` so the model learns where to stop.
- **Collator**: pads each batch of 64 to its own longest sequence — ids with
  the pad token, labels with -100, attention with 0.

## LoRA — the rank-16 bottleneck

The frozen model stays frozen. For each attention matrix (`q_proj, k_proj,
v_proj, o_proj`, in every one of the 24 layers) LoRA bolts on a trainable
detour:

```
            frozen W (e.g. 896×896, 802,816 weights)  🧊
  input ────────────────────────────────────────►(+)──► output
     └──► A (896→16) ──► B (16→896) ─────────────┘
          └──── the rank-16 bottleneck: trains 🔥 ────┘
```

Everything the fine-tune "learns" must squeeze through 16 numbers on its way
around each frozen matrix. `r=16` is a dial: enough capacity for a 6-rule
world, tiny cost. `lora_alpha=32` (2×r, convention) scales the detour's
contribution; dropout 0.05 is light regularization.

**Where 2,162,688 comes from** (the number train.py printed):

| matrix | frozen shape | A | B | trainable |
|---|---|---|---|---|
| q_proj | 896×896 | 896×16=14,336 | 16×896=14,336 | 28,672 |
| k_proj | 896×128 | 896×16=14,336 | 16×128=2,048 | 16,384 |
| v_proj | 896×128 | 896×16=14,336 | 16×128=2,048 | 16,384 |
| o_proj | 896×896 | 896×16=14,336 | 16×896=14,336 | 28,672 |
| **per layer** | | | | **90,112** |
| **× 24 layers** | | | | **2,162,688 ✓** |

2,162,688 ÷ 496,195,456 total = **0.44% trains**. That's why training took
10 minutes, fit easily in 24GB, and why the saved adapter is 8.7MB instead of
a 1GB model copy. (k/v are 896×128 because Qwen uses grouped-query attention:
2 KV heads × 64 dims.)

## The dials, and why these values

| dial | value | why |
|---|---|---|
| epochs | 3 | three full passes over 35k rows |
| learning rate | 2e-4 | high by pretraining standards, normal for LoRA — only adapters move |
| batch size | 64 | was 8 → GPU sat 78% idle on launch overhead; 64 gave 13× throughput |
| grad accum | 1 | not needed once batch 64 fits (drop to 8 + accum 2 on a T4/MPS) |
| max_len | 512 | safety cap; real sequences are ~81–120 tokens so it never bites |
| fp16 | on CUDA | half precision: 2× memory and speed, plenty for LoRA |

Step math: ceil(35,000 / 64) ≈ 547 steps/epoch × 3 = **1,641 steps**.

## The loop and the loss story

`trainer.train()` runs the classic four-beat 1,641 times: collate a batch →
forward pass + cross-entropy on unmasked positions → backward into the A/B
matrices only → optimizer step.

Loss went ~10 → 0.003 → 0.0. For a **deterministic** world with **one
canonical spelling** per state, near-zero loss is the correct outcome, not
overfitting-to-worry-about: there is exactly one right answer per position and
the model has capacity to learn the entire rulebook. (The honest test of
generalization is eval on unseen episodes — doc 06.)

## What gets saved

`save_pretrained` writes **only the adapter** (A/B matrices + config +
tokenizer) to `checkpoints/qwen05b-lora/`. It's a patch, not a model:
eval.py must load base Qwen first, then graft the adapter on with
`PeftModel.from_pretrained`.
