"""Intervention tool: change the numbers in a head's question and see what breaks.

Pick a station and a head. The tool zeroes that head's QUESTION (its 64 slots
of the query projection output) during a live run, then shows you:
  - the head's attention row for the last token, before vs after
  - the model's final top-3 bet, before vs after

Zeroed question => match scores all ~equal => the head's attention goes flat
(it listens to everyone equally instead of whoever it used to pick).

Usage:  python patch.py     then TEXT> , STATION> (1-24), HEAD> (0-13)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    print("loading base model (CPU)...")
    tok = AutoTokenizer.from_pretrained("checkpoints/qwen05b-lora")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", attn_implementation="eager", dtype=torch.float32)
    model.eval()
    print("ready. empty TEXT quits.\n")

    while True:
        try:
            text = input("TEXT>    ")
            if not text.strip():
                break
            st = int(input("STATION> ")) - 1
            hd = int(input("HEAD>    "))
        except (EOFError, KeyboardInterrupt):
            print()
            break
        except ValueError:
            print("  station must be 1-24, head 0-13")
            continue
        if not (0 <= st < 24 and 0 <= hd < 14):
            print("  station must be 1-24, head 0-13")
            continue

        ids = tok(text, return_tensors="pt")["input_ids"]
        pieces = [tok.decode([i]) for i in ids[0].tolist()]
        n = len(pieces)

        def run(zero):
            handle = None
            if zero:
                def hook(mod, inp, out):
                    out = out.clone()
                    out[..., hd * 64:(hd + 1) * 64] = 0.0
                    return out
                handle = model.model.layers[st].self_attn.q_proj.register_forward_hook(hook)
            with torch.no_grad():
                o = model(ids, output_attentions=True)
            if handle:
                handle.remove()
            probs = torch.softmax(o.logits[0, -1], -1)
            attn_row = o.attentions[st][0, hd, -1]
            return probs, attn_row

        p0, a0 = run(zero=False)
        p1, a1 = run(zero=True)

        print(f"\n  station {st+1}, head {hd} -- attention of last token {pieces[-1]!r}:")
        show = min(n, 10)
        print("    normal:  " + "  ".join(f"{pieces[j]!r} {a0[j]*100:.0f}%" for j in range(n - show, n)))
        print("    zeroed:  " + "  ".join(f"{pieces[j]!r} {a1[j]*100:.0f}%" for j in range(n - show, n)))

        t0 = torch.topk(p0, 3)
        t1 = torch.topk(p1, 3)
        print("  final bet:")
        print("    normal:  " + "  ".join(f"{tok.decode([i])!r} {v*100:.1f}%" for v, i in zip(t0.values, t0.indices)))
        print("    zeroed:  " + "  ".join(f"{tok.decode([i])!r} {v*100:.1f}%" for v, i in zip(t1.values, t1.indices)))
        shift = (p0 - p1).abs().sum().item() / 2
        print(f"  total probability moved by silencing this one question: {shift*100:.2f}%\n")


if __name__ == "__main__":
    main()
