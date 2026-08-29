"""Deterministic 5x5 gridworld with a canonical text serializer.

Coordinates are (x, y), 0-indexed, with (0, 0) at the south-west corner:
north = y+1, south = y-1, east = x+1, west = x-1.

Rules:
- Moves into a grid edge, a wall, or a locked door cell are no-ops.
- "pick up" succeeds only when the agent stands on the key cell.
- "unlock" succeeds only when the agent holds the key, the door is locked,
  and the agent is orthogonally adjacent to the door. The key is kept.
- Every invalid action leaves the state unchanged (the env never errors).
"""

from dataclasses import dataclass, replace
import random

GRID_SIZE = 5

ACTIONS = ["move north", "move south", "move east", "move west", "pick up", "unlock"]

_DELTAS = {
    "move north": (0, 1),
    "move south": (0, -1),
    "move east": (1, 0),
    "move west": (-1, 0),
}

PROMPT_TEMPLATE = "STATE: {state}\nACTION: {action}\nNEXT: "


@dataclass(frozen=True)
class State:
    agent: tuple          # (x, y)
    key: tuple            # (x, y) or None once held
    door: tuple           # (x, y)
    locked: bool
    has_key: bool
    walls: tuple          # sorted tuple of (x, y)


def serialize(s: State) -> str:
    key_part = "held" if s.key is None else f"({s.key[0]},{s.key[1]})"
    walls_part = ",".join(f"({x},{y})" for x, y in s.walls)
    return (
        f"agent=({s.agent[0]},{s.agent[1]}); "
        f"key={key_part}; "
        f"door=({s.door[0]},{s.door[1]}) {'locked' if s.locked else 'unlocked'}; "
        f"has_key={'true' if s.has_key else 'false'}; "
        f"walls=[{walls_part}]"
    )


def step(s: State, action: str) -> State:
    if action in _DELTAS:
        dx, dy = _DELTAS[action]
        nx, ny = s.agent[0] + dx, s.agent[1] + dy
        target = (nx, ny)
        blocked = (
            not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE)
            or target in s.walls
            or (target == s.door and s.locked)
        )
        return s if blocked else replace(s, agent=target)
    if action == "pick up":
        if not s.has_key and s.key == s.agent:
            return replace(s, key=None, has_key=True)
        return s
    if action == "unlock":
        adjacent = abs(s.agent[0] - s.door[0]) + abs(s.agent[1] - s.door[1]) == 1
        if s.has_key and s.locked and adjacent:
            return replace(s, locked=False)
        return s
    raise ValueError(f"unknown action: {action}")


def random_state(rng: random.Random) -> State:
    cells = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
    rng.shuffle(cells)
    n_walls = rng.randint(0, 2)
    walls = tuple(sorted(cells[:n_walls]))
    door, key, agent = cells[n_walls], cells[n_walls + 1], cells[n_walls + 2]
    return State(agent=agent, key=key, door=door, locked=True, has_key=False, walls=walls)


if __name__ == "__main__":
    rng = random.Random(0)
    s = random_state(rng)
    print(serialize(s))
    for a in ["move north", "pick up", "unlock", "move east"]:
        s = step(s, a)
        print(f"{a:12s} -> {serialize(s)}")
