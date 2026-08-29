"""Autopsy of the trained world model: find, save, and classify every miss.

1. Single-step: ALL 3,000 held-out transitions (eval.py sampled 500).
2. Rollout: all 150 held-out episodes, 10 steps, model output fed back in.

Every misprediction is written to autopsy_misses.jsonl with a field-level
diagnosis (which of agent/key/door/has_key/walls is wrong, and how). A
human-readable summary lands in autopsy_report.txt. Progress is printed
continuously so a tailed log shows the process live.
"""

import json
import time
from collections import Counter, defaultdict

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from env import PROMPT_TEMPLATE

BASE = "Qwen/Qwen2.5-0.5B"
ADAPTER = "checkpoints/qwen05b-lora"
TEST = "data/test.jsonl"
BATCH = 32
HORIZON = 10
FIELDS = ["agent", "key", "door", "has_key", "walls"]

log_t0 = time.time()
def log(msg):
    print(f"[{time.time() - log_t0:7.1f}s] {msg}", flush=True)


def split_fields(s):
    parts = s.split("; ")
    return parts if len(parts) == 5 else None


def diagnose(truth, pred, prev):
    """Which fields are wrong, and what *kind* of mistake is it?"""
    pt, pp = split_fields(truth), split_fields(pred)
    if pp is None:
        return ["FORMAT"], "malformed output (not 5 fields)"
    wrong = [FIELDS[i] for i in range(5) if pt[i] != pp[i]]
    should_change = truth != prev
    did_change = pred != prev
    if not should_change and did_change:
        kind = "phantom change (was a no-op, model changed something)"
    elif should_change and not did_change:
        kind = "missed change (model predicted no-op)"
    elif should_change and did_change:
        kind = "wrong change (changed, but incorrectly)"
    else:
        kind = "impossible"
    return wrong, kind


@torch.no_grad()
def predict_batch(model, tok, device, states, actions):
    prompts = [PROMPT_TEMPLATE.format(state=s, action=a)
               for s, a in zip(states, actions)]
    enc = tok(prompts, return_tensors="pt", padding=True,
              padding_side="left").to(device)
    out = model.generate(**enc, max_new_tokens=160, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    comps = tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                             skip_special_tokens=True)
    return [c.split("\n")[0].strip() for c in comps]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"loading base model {BASE} on {device}...")
    # Base tokenizer == the one saved with the adapter (train.py never modified
    # it); loading from BASE avoids shipping 14MB of tokenizer files to the pod.
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    model = PeftModel.from_pretrained(model, ADAPTER).to(device).eval()
    log("model + adapter loaded.")

    rows = [json.loads(l) for l in open(TEST)]
    episodes = defaultdict(list)
    for r in rows:
        episodes[r["episode"]].append(r)
    for ep in episodes.values():
        ep.sort(key=lambda r: r["step"])
    log(f"test set: {len(rows)} transitions, {len(episodes)} episodes")

    misses = []

    # ---------- phase 1: single-step over ALL 3,000 ----------
    log("=== PHASE 1: single-step, all 3,000 transitions ===")
    correct = 0
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i + BATCH]
        preds = predict_batch(model, tok, device,
                              [r["state"] for r in chunk],
                              [r["action"] for r in chunk])
        for r, pred in zip(chunk, preds):
            if pred == r["next_state"]:
                correct += 1
            else:
                wrong, kind = diagnose(r["next_state"], pred, r["state"])
                misses.append({"phase": "single", "episode": r["episode"],
                               "step": r["step"], "action": r["action"],
                               "state": r["state"], "truth": r["next_state"],
                               "pred": pred, "wrong_fields": wrong, "kind": kind})
                log(f"  MISS #{len(misses)} ep{r['episode']} step{r['step']} "
                    f"action='{r['action']}' fields={wrong} ({kind})")
        done = i + len(chunk)
        if done % (BATCH * 10) == 0 or done == len(rows):
            log(f"  progress {done}/{len(rows)}  "
                f"acc so far {correct}/{done} = {correct / done:.2%}")
    single_misses = [m for m in misses if m["phase"] == "single"]
    log(f"phase 1 done: {correct}/{len(rows)} = {correct / len(rows):.2%} "
        f"exact  ({len(single_misses)} misses)")

    # ---------- phase 2: rollouts, all 150 episodes ----------
    log(f"=== PHASE 2: {HORIZON}-step rollouts, all {len(episodes)} episodes ===")
    eps = list(episodes.values())
    survived = 0
    for i in range(0, len(eps), BATCH):
        batch = eps[i:i + BATCH]
        cur = [ep[0]["state"] for ep in batch]
        dead = [False] * len(batch)
        for t in range(HORIZON):
            actions = [ep[t]["action"] for ep in batch]
            prev = cur
            cur = predict_batch(model, tok, device, cur, actions)
            for j, ep in enumerate(batch):
                if not dead[j] and cur[j] != ep[t]["next_state"]:
                    dead[j] = True
                    wrong, kind = diagnose(ep[t]["next_state"], cur[j], prev[j])
                    misses.append({"phase": "rollout", "episode": ep[0]["episode"],
                                   "diverged_at_step": t + 1,
                                   "action": ep[t]["action"], "state": prev[j],
                                   "truth": ep[t]["next_state"], "pred": cur[j],
                                   "wrong_fields": wrong, "kind": kind})
                    log(f"  DIVERGED ep{ep[0]['episode']} at step {t + 1} "
                        f"action='{ep[t]['action']}' fields={wrong} ({kind})")
        survived += sum(1 for d in dead if not d)
        log(f"  progress {min(i + BATCH, len(eps))}/{len(eps)} episodes  "
            f"(clean so far: {survived})")
    log(f"phase 2 done: {survived}/{len(eps)} episodes exact through "
        f"all {HORIZON} steps")

    # ---------- report ----------
    with open("autopsy_misses.jsonl", "w") as f:
        for m in misses:
            f.write(json.dumps(m) + "\n")

    lines = []
    def rep(s=""):
        lines.append(s)
        log(s if s else "-")
    rep(f"AUTOPSY REPORT  ({time.strftime('%Y-%m-%d %H:%M')})")
    rep(f"single-step: {correct}/{len(rows)} = {correct / len(rows):.2%} "
        f"({len(single_misses)} misses)")
    rep(f"rollout: {survived}/{len(eps)} episodes clean through {HORIZON} steps")
    rep()
    rep("misses by action:")
    for a, n in Counter(m["action"] for m in misses).most_common():
        rep(f"  {a:12s} {n}")
    rep("misses by wrong field(s):")
    for wf, n in Counter(tuple(m["wrong_fields"]) for m in misses).most_common():
        rep(f"  {'+'.join(wf):20s} {n}")
    rep("misses by kind:")
    for k, n in Counter(m["kind"] for m in misses).most_common():
        rep(f"  {k:55s} {n}")
    rep()
    rep("first 5 misses in full:")
    for m in misses[:5]:
        rep(f"  [{m['phase']}] ep{m['episode']} action='{m['action']}'")
        rep(f"    state: {m['state']}")
        rep(f"    truth: {m['truth']}")
        rep(f"    pred : {m['pred']}")
    open("autopsy_report.txt", "w").write("\n".join(lines) + "\n")
    log("wrote autopsy_misses.jsonl and autopsy_report.txt")


if __name__ == "__main__":
    main()
