"""Raw continuation REPL for the UNTUNED base model (no chat template, no wrapper).

You type any text; base Qwen2.5-0.5B continues it token by token, greedily,
exactly as pretrained -- a text-continuer, not an assistant.

Usage:
    python complete.py                 # 20 tokens per turn, plain output
    python complete.py --steps 40      # longer continuations
    python complete.py --peek          # show top-3 candidate pages per step
    python complete.py --sample        # sample from softmax instead of greedy
                                       # (temperature 0.8; re-run same prompt
                                       #  to watch answers vary)

Runs on CPU (~2 GB, a few seconds to load, ~1s/token). Ctrl-C or empty line quits.
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20, help="tokens to generate per prompt")
    p.add_argument("--peek", action="store_true", help="print top-3 pages + %% each step")
    p.add_argument("--sample", action="store_true", help="sample (T=0.8) instead of greedy")
    args = p.parse_args()

    print("loading base model (CPU)...")
    tok = AutoTokenizer.from_pretrained("checkpoints/qwen05b-lora")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.float32)
    model.eval()
    print("ready. type text; empty line quits.\n")

    while True:
        try:
            text = input("TEXT> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text.strip():
            break

        ids = tok(text, return_tensors="pt")["input_ids"]
        print(f"  [{ids.shape[1]} prompt tokens]")
        print(text, end="", flush=True)

        past = None
        for step in range(args.steps):
            with torch.no_grad():
                out = model(ids if past is None else ids[:, -1:],
                            past_key_values=past, use_cache=True)
            past = out.past_key_values
            probs = torch.softmax(out.logits[0, -1], dim=-1)

            if args.sample:
                pick = torch.multinomial(torch.softmax(out.logits[0, -1] / 0.8, -1), 1)[0]
            else:
                pick = probs.argmax()

            if args.peek:
                top = torch.topk(probs, 3)
                cands = "  ".join(f"{tok.decode([i])!r} {v * 100:.0f}%"
                                  for v, i in zip(top.values, top.indices))
                print(f"\n    step {step + 1:2d}: {cands}", end="")
                print(f"\n{tok.decode(ids[0])}{tok.decode([pick])}", end="", flush=True)
            else:
                print(tok.decode([pick]), end="", flush=True)

            ids = torch.cat([ids, pick.view(1, 1)], dim=1)
            if pick == tok.eos_token_id:
                break
        print("\n")


if __name__ == "__main__":
    main()
