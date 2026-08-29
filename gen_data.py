"""Roll out lightly-biased random episodes and write train/test JSONL.

Split is by episode, not by transition, so multi-step eval is honest.
Each line: {"episode": int, "step": int, "state": str, "action": str, "next_state": str}
"""

import argparse
import json
import os
import random

from env import ACTIONS, random_state, serialize, step


def _toward(src, dst, rng):
    moves = []
    if dst[0] > src[0]:
        moves.append("move east")
    if dst[0] < src[0]:
        moves.append("move west")
    if dst[1] > src[1]:
        moves.append("move north")
    if dst[1] < src[1]:
        moves.append("move south")
    return rng.choice(moves) if moves else None


def choose_action(s, rng):
    # Bias toward the rare interactions (and toward reaching them) so
    # successful pick-up/unlock transitions are well represented.
    if not s.has_key and s.key == s.agent and rng.random() < 0.6:
        return "pick up"
    adjacent = abs(s.agent[0] - s.door[0]) + abs(s.agent[1] - s.door[1]) == 1
    if s.has_key and s.locked and adjacent and rng.random() < 0.6:
        return "unlock"
    if rng.random() < 0.35:
        goal = s.key if not s.has_key else (s.door if s.locked else None)
        if goal is not None:
            move = _toward(s.agent, goal, rng)
            if move is not None:
                return move
    return rng.choice(ACTIONS)


def rollout(episode_id, length, rng):
    s = random_state(rng)
    transitions = []
    for t in range(length):
        a = choose_action(s, rng)
        s2 = step(s, a)
        transitions.append(
            {
                "episode": episode_id,
                "step": t,
                "state": serialize(s),
                "action": a,
                "next_state": serialize(s2),
            }
        )
        s = s2
    return transitions


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-episodes", type=int, default=1750)
    p.add_argument("--test-episodes", type=int, default=150)
    p.add_argument("--episode-length", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="data")
    args = p.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    train, test = [], []
    total = args.train_episodes + args.test_episodes
    for ep in range(total):
        rows = rollout(ep, args.episode_length, rng)
        (train if ep < args.train_episodes else test).extend(rows)

    write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train)
    write_jsonl(os.path.join(args.out_dir, "test.jsonl"), test)

    changed = sum(r["state"] != r["next_state"] for r in train)
    by_action = {}
    for r in train:
        by_action.setdefault(r["action"], [0, 0])
        by_action[r["action"]][0] += 1
        by_action[r["action"]][1] += r["state"] != r["next_state"]
    print(f"train: {len(train)} transitions ({args.train_episodes} episodes)")
    print(f"test:  {len(test)} transitions ({args.test_episodes} episodes)")
    print(f"state changed in {changed}/{len(train)} train transitions")
    for a in sorted(by_action):
        n, c = by_action[a]
        print(f"  {a:12s} n={n:6d}  changed={c:6d} ({c / n:.0%})")


if __name__ == "__main__":
    main()
