"""Evaluate the trained next-state predictor.

1. Single-step exact-match accuracy on held-out transitions, vs. the trivial
   "state unchanged" baseline.
2. Multi-step rollout drift: starting from each test episode's true initial
   state, feed the model's own predictions back in for --horizon steps (using
   the episode's true action sequence) and record per-step match vs. the env.

Writes eval_drift.png if matplotlib is installed; always prints the table.
"""

import argparse
import json
from collections import defaultdict

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from env import PROMPT_TEMPLATE


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def predict_batch(model, tokenizer, device, states, actions, max_new_tokens=160):
    prompts = [
        PROMPT_TEMPLATE.format(state=s, action=a) for s, a in zip(states, actions)
    ]
    enc = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").to(device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    completions = tokenizer.batch_decode(
        out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return [c.split("\n")[0].strip() for c in completions]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--adapter", default="checkpoints/qwen05b-lora")
    p.add_argument("--test-file", default="data/test.jsonl")
    p.add_argument("--single-step-n", type=int, default=500)
    p.add_argument("--rollout-episodes", type=int, default=50)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model = PeftModel.from_pretrained(model, args.adapter).to(device).eval()

    rows = load_jsonl(args.test_file)
    episodes = defaultdict(list)
    for r in rows:
        episodes[r["episode"]].append(r)
    for ep in episodes.values():
        ep.sort(key=lambda r: r["step"])

    # --- single-step accuracy ---
    sample = rows[: args.single_step_n]
    correct = baseline_correct = 0
    for i in range(0, len(sample), args.batch_size):
        chunk = sample[i : i + args.batch_size]
        preds = predict_batch(
            model, tokenizer, device,
            [r["state"] for r in chunk], [r["action"] for r in chunk],
        )
        for r, pred in zip(chunk, preds):
            correct += pred == r["next_state"]
            baseline_correct += r["state"] == r["next_state"]
    n = len(sample)
    print(f"single-step exact match: {correct}/{n} = {correct / n:.1%}")
    print(f"'unchanged' baseline:    {baseline_correct}/{n} = {baseline_correct / n:.1%}")

    # --- multi-step rollout drift ---
    eps = list(episodes.values())[: args.rollout_episodes]
    match_at_step = [0] * args.horizon
    first_divergence = []
    for i in range(0, len(eps), args.batch_size):
        batch = eps[i : i + args.batch_size]
        cur = [ep[0]["state"] for ep in batch]  # model's own belief state
        diverged_at = [None] * len(batch)
        for t in range(args.horizon):
            actions = [ep[t]["action"] for ep in batch]
            cur = predict_batch(model, tokenizer, device, cur, actions)
            for j, ep in enumerate(batch):
                if cur[j] == ep[t]["next_state"]:
                    match_at_step[t] += 1
                elif diverged_at[j] is None:
                    diverged_at[j] = t + 1
        first_divergence.extend(d if d is not None else args.horizon + 1 for d in diverged_at)

    total = len(eps)
    print(f"\nrollout drift over {total} episodes (model predictions fed back in):")
    print("step  match-rate")
    rates = []
    for t in range(args.horizon):
        rate = match_at_step[t] / total
        rates.append(rate)
        print(f"{t + 1:4d}  {rate:.1%}")
    never = sum(d > args.horizon for d in first_divergence)
    print(f"\nepisodes still exact after {args.horizon} steps: {never}/{total}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        plt.plot(range(1, args.horizon + 1), rates, marker="o")
        plt.xlabel("rollout step")
        plt.ylabel("exact-match rate vs. true env")
        plt.title("World-model drift")
        plt.ylim(0, 1.05)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("eval_drift.png", dpi=150)
        print("wrote eval_drift.png")
    except ImportError:
        print("matplotlib not installed; skipped plot")


if __name__ == "__main__":
    main()
