"""Interactive REPL: hand-type a state + action, see what the model dreams.

Run ON the pod (needs the GPU):
  cd /workspace/sim-test && source /workspace/venv/bin/activate
  export HF_HOME=/workspace/hf && python dream.py

Commands at the STATE prompt:
  <a full state string>   use exactly what you typed
  r                       deal a fresh random state (env.random_state)
  <enter>                 reuse the last state (chain your own rollout!)
  q                       quit

After each prediction the true env answer is shown too, so every prompt you
type is instantly graded MATCH / MISMATCH.
"""

import random
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from env import ACTIONS, PROMPT_TEMPLATE, State, random_state, serialize, step

BASE = "Qwen/Qwen2.5-0.5B"
ADAPTER = "checkpoints/qwen05b-lora"

RX = re.compile(
    r"agent=\((\d),(\d)\); key=(held|\((\d),(\d)\)); door=\((\d),(\d)\) "
    r"(locked|unlocked); has_key=(true|false); walls=\[(.*)\]"
)


def parse_state(s):
    """State string -> env.State, or None if it doesn't match the format."""
    m = RX.fullmatch(s.strip())
    if not m:
        return None
    key = None if m.group(3) == "held" else (int(m.group(4)), int(m.group(5)))
    walls = tuple(sorted(
        (int(x), int(y)) for x, y in re.findall(r"\((\d),(\d)\)", m.group(10))))
    return State(
        agent=(int(m.group(1)), int(m.group(2))),
        key=key,
        door=(int(m.group(6)), int(m.group(7))),
        locked=m.group(8) == "locked",
        has_key=m.group(9) == "true",
        walls=walls,
    )


def draw(s):
    """Tiny board so you can see what you typed."""
    for y in range(4, -1, -1):
        row = []
        for x in range(5):
            c = "."
            if (x, y) in s.walls: c = "#"
            if s.key == (x, y): c = "k"
            if s.door == (x, y): c = "D" if s.locked else "d"
            if s.agent == (x, y): c = "A"
            row.append(c)
        print("   " + " ".join(row))


@torch.no_grad()
def predict(model, tok, device, state_str, action):
    prompt = PROMPT_TEMPLATE.format(state=state_str, action=action)
    enc = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**enc, max_new_tokens=160, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    comp = tok.decode(out[0, enc["input_ids"].shape[1]:],
                      skip_special_tokens=True)
    return comp.split("\n")[0].strip()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {BASE} + adapter on {device} (~30s)...")
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    model = PeftModel.from_pretrained(model, ADAPTER).to(device).eval()
    print("ready. actions:", ", ".join(ACTIONS))
    print(__doc__.split("Commands")[1].split("After")[0])

    rng = random.Random()
    last = None
    while True:
        try:
            raw = input("\nSTATE (r=random, enter=reuse last, q=quit)> ").strip()
        except EOFError:
            break
        if raw == "q":
            break
        if raw == "r":
            s = random_state(rng)
        elif raw == "" and last is not None:
            s = last
        else:
            s = parse_state(raw)
            if s is None:
                print("  couldn't parse that — format must be exactly like:")
                print("  agent=(2,3); key=(0,1); door=(4,4) locked; "
                      "has_key=false; walls=[(1,1),(3,2)]")
                continue
        state_str = serialize(s)
        print("  " + state_str)
        draw(s)

        action = input("ACTION> ").strip()
        if action not in ACTIONS:
            print(f"  unknown action (choose from: {', '.join(ACTIONS)})")
            continue

        model_out = predict(model, tok, device, state_str, action)
        truth = serialize(step(s, action))
        verdict = "MATCH ✓" if model_out == truth else "MISMATCH ✗"
        print(f"  MODEL: {model_out}")
        print(f"  ENV  : {truth}")
        print(f"  --> {verdict}")

        t = parse_state(model_out)
        if t is None:
            print("  (model output is not even a valid state string!)")
            last = s
        else:
            last = t  # chain: next <enter> feeds the MODEL's belief back in
            if verdict.startswith("MISMATCH"):
                print("  (note: pressing enter next continues from the "
                      "MODEL's wrong belief — a live drift experiment)")


if __name__ == "__main__":
    main()
