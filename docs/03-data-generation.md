# 03 · gen_data.py — where the 35,000 rows come from

One loop: **deal a world, press 20 buttons, write down what the physics said.**
Repeated 1,900 times from a single seeded RNG.

## The contract with env.py

```
gen_data.py (player + scribe)              env.py (world + laws)
─────────────────────────────              ─────────────────────
rollout():  "deal me a world"    ───────►  random_state(rng)
  20×:  choose_action(s)           (only decision gen_data makes itself)
        "what happens if…"       ───────►  step(s, action)
        "write both down"        ───────►  serialize(s), serialize(s2)
        s = s2                     (output becomes next input → a trajectory)
```

`gen_data.py` never computes physics. It asks. So the dataset cannot
contradict the rules.

## choose_action() — the four-personality player

Checked in priority order; every threshold is a weighted coin
(`rng.random() < p` is True with probability p):

1. **Reflex, pick up (60%)** — standing on the key, not holding it.
2. **Reflex, unlock (60%)** — key in hand, door locked, adjacent.
3. **Compass (35%)** — step toward the current goal: the key if unheld,
   else the door if locked, else nothing. Wall-blind on purpose.
4. **Dice** — uniform over all six actions, including illegal ones.

Across the real dataset: dice 23,604 choices · compass 9,460 ·
pick-up reflex 1,174 · unlock reflex 762.

Why the biases exist: pure dice produced only **101 successful unlocks in
35,000 rows** — statistically invisible, so the model couldn't learn the
rarest rule. With biases: 1,314 successful pick-ups, 813 successful unlocks.
Why they're not 100%: the data must also contain "stood on the key and walked
away" and wall-bumps — wrong behavior is coverage.

## Episode ids — name tags, not quantities

`"episode": N` is just the loop counter stamped on each of that episode's 20
rows. It is **never shown to the model**. Its three jobs:

1. **The split** — `ep < 1750` → train.jsonl, else test.jsonl. Whole episodes,
   never split, so test worlds are truly unseen (episodes share a layout
   across their 20 rows — splitting by row would leak layouts).
2. **Regrouping** — eval.py collects rows by id, sorts by `step`, and replays
   them for the 10-step rollout test.
3. **Archaeology** — how we looked up "episode 720" (below).

Arithmetic, not multiplication of ids: 1,750 × 20 = 35,000 train rows;
150 × 20 = 3,000 test rows.

## Why the agent sometimes never moves (episode 720)

**35.4% of all rows (12,386) are no-ops** — the policy pressed an illegal
button and `step()` answered "nothing happens." Breakdown across the dataset:
edge bumps 2,405 · keyless unlocks 2,197 · off-key pick-ups 2,080 ·
already-held pick-ups 1,697 · wall bumps 1,230 · locked-door bumps 1,089 ·
unlocking open doors 926 · unlocking from too far 762.

The extreme case, **episode 720: 20 steps, zero movement.** The spawn:

```
 4   .   .   .   .   .        agent (4,1) is in a pocket:
 3   .   .  D🔒  .   .          west  → wall (3,1)
 2   .   .   .   .   #          north → wall (4,2)
 1   .  🗝️  .   #   A          east  → grid edge
 0   .   .   .   .   .          south → OPEN … but never rolled
     0   1   2   3   4
```

The key sits due **west**, so the wall-blind compass kept marching the agent
into the wall at (3,1) — 10 of the 20 presses. The dice hit east (edge),
north (wall), and pick up (not on key), but never south. Twenty presses,
twenty no-ops — and all twenty are **valid training rows**: the correct
prediction each time is the input string unchanged.

That's the lesson: "when nothing happens" is a third of the physics, and it's
exactly why the trivial "predict no change" baseline scores 42.6%.

## Determinism

One `random.Random(42)` threads through everything. Same seed → byte-identical
dataset (we verified this by replaying episodes 0 and 720 and diffing against
the real files). Caveat: the number of coin flips consumed per step depends on
the state (short-circuit `and`), so any replay must replicate the branch
structure exactly or the stream desynchronizes.
