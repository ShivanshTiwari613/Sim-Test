# 01 · The big picture

## What is a "world model"?

A world model answers one question: **given how things are and what I do,
what happens next?** Formally, it's a learned function

```
f(state_t, action_t) → state_t+1
```

Big systems (Ha & Schmidhuber's World Models, the Dreamer papers) learn `f`
in a compressed latent space from pixels. This project builds the smallest
honest version: the state is a **line of text**, and `f` is a small LLM
fine-tuned to complete that text.

```
STATE: agent=(2,3); key=(0,1); door=(4,4) locked; has_key=false; walls=[(1,1)]
ACTION: move north
NEXT: agent=(2,4); key=(0,1); door=(4,4) locked; has_key=false; walls=[(1,1)]
       └──────────── the model writes this line ────────────┘
```

The core design rule: **the state string is the memory.** No database, no
vector store, no hidden variables. Everything the model can know about the
world is in the prompt — so if it predicts correctly, its own output is a
complete input for the next step. That's what makes multi-step "dreaming"
possible.

## The whole pipeline

```
env.py          the world: rules + canonical text format
   │
gen_data.py     play 1,900 episodes → 35k train / 3k test text triples
   │
train.py        fine-tune Qwen2.5-0.5B with LoRA (10 min on a 4090)
   │
eval.py         grade it: single-step exact match + 10-step rollout drift
```

## What happened when we ran it (2026-08-25)

| Metric | Result |
|---|---|
| Single-step exact match | **98.0%** (baseline "predict no change": 42.6%) |
| Rollout, step 10, own predictions fed back | **96.0%** still exact |
| Episodes perfect through all 10 steps | 48 / 50 |
| Training time / cost | 10 min on an RTX 4090 / ≈ $0.15 |

The headline is the **flat drift curve**: 98% → 96% over ten self-fed steps.
Errors don't compound, because a correctly predicted state is a fully
self-contained input for the next prediction. The 2 episodes that slipped
diverged once and stayed diverged; the other 48 stayed byte-perfect.

## Why this is worth building before reading the papers

Every concept in Dreamer maps onto a part you can now point at:

| Here (text world) | There (latent world models) |
|---|---|
| canonical state string | learned latent vector z |
| Qwen + LoRA next-state predictor | RSSM / dynamics network |
| feeding predictions back for 10 steps | "imagination" rollouts |
| exact-match drift curve | reconstruction / prediction error |

Same loop, different representation.
