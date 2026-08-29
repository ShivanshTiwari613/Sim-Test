# 06 · eval.py — how we know it worked

Two tests, both on the 150 held-out episodes (worlds the model never saw,
split by episode id so nothing leaks). Grading is brutal on purpose:
**string equality**. One wrong character anywhere = wrong.

## Test 1 — single-step exact match

Take 500 held-out transitions, feed each `STATE/ACTION` prompt, generate
greedily (temperature 0 — a physics engine shouldn't be creative), compare to
the true next state:

```
model:                 490 / 500 = 98.0%
"unchanged" baseline:  213 / 500 = 42.6%
```

The baseline is the whole reason the number means something. A model could
score ~43% by literally copying its input — because 35–43% of transitions
genuinely are no-ops. 98% vs 42.6% shows it learned *dynamics*, not copying.

## Test 2 — rollout drift (the real point)

The world-model test: can it **dream**?

```
true start ──► model predicts s₁ ──► feed s₁ back ──► predicts s₂ ──► … ×10
                     │ compare              │ compare
true start ──► env: true s₁    ──►     env: true s₂  ──► …
```

The model's own output becomes its next input — the true environment is only
used as the answer key. Actions come from the episode's recorded sequence.
This only works because of two earlier decisions: the output format equals
the input format (same `serialize`, same `PROMPT_TEMPLATE` constant), and the
state string is complete — no hidden memory to lose between steps.

Result over 50 episodes:

```
step:   1    2    3    4    5    6    7    8    9    10
exact: 98%  98%  98%  98%  98%  98%  96%  96%  96%  96%
```

**The flatness is the finding.** Errors don't compound: 48/50 episodes stayed
byte-perfect through all ten steps; the 2 that slipped diverged once and
stayed diverged. A correct predicted state is a fully self-contained input,
so correctness is self-sustaining — the payoff of "the string is the memory."

(`eval_drift.png` in the repo root is this curve, drawn by eval.py on the pod.)

## Mechanics worth knowing

- Episode ids do their second job here: rows are grouped by id and sorted by
  `step` to reconstruct each trajectory.
- Generation stops at `<|endoftext|>` / first newline — the trained stop sign.
- Batched generation uses left-padding (completion must sit flush against the
  generation boundary).
- Open thread: nobody has yet inspected *which* rule the 2 diverging episodes
  broke — the natural next experiment.
