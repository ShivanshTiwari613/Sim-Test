# 02 · env.py — the world and its laws

`env.py` is ~100 lines and contains the entire universe: the state shape, the
physics (`step`), the scribe (`serialize`), and the dealer (`random_state`).

## The problem it solves (LeetCode style)

> **Design a Gridworld Simulator.** A 5×5 grid holds one agent, one key, one
> door (initially locked), and 0–2 walls. Coordinates are (x, y), 0-indexed,
> **(0,0) at the bottom-left**; north = y+1. Implement:
>
> `step(state, action) → state` for six actions:
> - `move north/south/east/west` — fails if target is off-grid, a wall, or
>   the door while locked. An **unlocked door cell is walkable**.
> - `pick up` — only while standing **on** the key; sets `key=held` and
>   `has_key=true` together.
> - `unlock` — only with key in hand, door locked, and agent **orthogonally
>   adjacent** (Manhattan distance exactly 1). Key is not consumed.
>
> ⚠️ A failed action is **not an error** — return the state unchanged.
> `step` is total: every (state, action) has exactly one answer.
>
> `serialize(state) → str` must be **canonical**: equal states ⇒
> byte-identical strings (fixed order, fixed spacing, sorted walls).

## step() — the physics, branch by branch

```python
if action in _DELTAS:                       # movement
    dx, dy = _DELTAS[action]                # {"move north": (0,1), ...}
    target = (agent.x + dx, agent.y + dy)   # vector addition
    blocked = ( off-grid  or  target in walls  or  (target == door and locked) )
    return s if blocked else replace(s, agent=target)
```

That one `blocked` boolean **is** the movement rulebook — the three and only
three things that stop you. The subtle clause: the door blocks *only while
locked*; once unlocked, clause C is False and the door behaves as floor. No
separate "walk through door" code exists.

```python
if action == "pick up":
    if not s.has_key and s.key == s.agent:
        return replace(s, key=None, has_key=True)   # two fields, one atomic change
    return s
```

```python
if action == "unlock":
    adjacent = |agent.x − door.x| + |agent.y − door.y| == 1   # Manhattan distance
    if s.has_key and s.locked and adjacent:
        return replace(s, locked=False)
    return s
```

```
 . ✓ .     Manhattan dist 1 → can unlock
 ✓ D ✓     diagonal (dist 2) → cannot
 . ✓ .     standing on it → impossible (locked doors block entry)
```

The final `raise ValueError` fires only on **unknown** action strings (typos) —
a programmer tripwire, not a game rule. Illegal-but-known actions return the
state unchanged; that's the source of every "agent stands still" row.

Key properties:
- `State` is a **frozen dataclass** → immutable. Every change is
  `replace(s, field=value)` — a copy with one surgical edit. Old states
  survive, which is what makes episodes recordable and replayable.
- Only two possible outcomes per call: same state back, or a copy with one
  field group changed.

## serialize() — the scribe

```
agent=(3,4); key=held; door=(1,4) locked; has_key=true; walls=[(2,2),(3,1)]
```

- Fixed field order, `"; "` separators, lowercase `true/false`, no spaces in
  tuples. Every rule removes one degree of formatting freedom.
- `key` is the only field that changes *shape*: `(x,y)` on the floor, the
  word `held` after pick-up. `key=held` + `has_key=true` is deliberate
  redundancy — a consistency trap that exact-match eval can catch.
- Walls are **not sorted here** — they were sorted at creation and the state
  is immutable, so canonical order is an invariant, not a chore.

Why so strict? The string is (1) the training label, (2) the eval judge
(`pred == truth`), and (3) the next step's input during rollouts. One state,
one spelling — or model capacity leaks into learning format noise.

## random_state() — the dealer

Shuffle-then-slice: shuffle all 25 cells once, then deal off the top —
first `randint(0,2)` cards become walls (sorted at birth), next three become
door, key, agent. Distinctness is guaranteed by construction (five cards from
one deck can't repeat) — zero collision checks needed. Every episode starts
`locked=True, has_key=False`: random *where*, fixed *quest*.

The `rng` is passed in, never global — one seeded stream drives the whole
dataset, which is why seed 42 reproduces all 38,000 rows byte-identically.
