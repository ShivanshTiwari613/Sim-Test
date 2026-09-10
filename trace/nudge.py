"""Watch one training nudge happen, by hand.

You type a prompt and a TARGET (what the model SHOULD say next). The script:
  1. runs the forward pass, shows p(target) and the current top-3,
  2. computes the loss -ln p(target) and runs backpropagation,
  3. shows the credit assignment: gradient traffic per station, per head,
     and which dictionary rows got blamed hardest,
  4. applies ONE gradient-descent step (a real nudge, in place), and
     re-runs the forward pass so you can see p(target) move.

Nudges accumulate until you type 'reset' (reloads factory weights) -- so you
can nudge the same prompt repeatedly and literally hand-train the model.

Usage:
    python nudge.py              # lr = 0.0003, gentle visible steps
    python nudge.py --lr 0.002   # violent nudges (one step can hit 90%+)

At the prompts:  TEXT>  any text.   TARGET>  the desired next word/piece
(only its first token is used; leading space matters: ' hello' != 'hello').
Empty TEXT quits; 'reset' at TEXT restores factory weights.
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BAR = "█"


def bars(vals, width=30):
    mx = max(vals) or 1.0
    return [BAR * max(1, round(v / mx * width)) for v in vals]


def load():
    m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.float32)
    m.train()  # no dropout in this model; just enables grad bookkeeping
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=0.0003, help="size of one nudge")
    args = ap.parse_args()

    print("loading base model (CPU)...")
    tok = AutoTokenizer.from_pretrained("checkpoints/qwen05b-lora")
    model = load()
    print("ready. TEXT then TARGET; empty TEXT quits; 'reset' restores factory weights.\n")

    while True:
        try:
            text = input("TEXT>   ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text.strip():
            break
        if text.strip() == "reset":
            print("reloading factory weights...")
            model = load()
            continue
        try:
            target = input("TARGET> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        tgt_ids = tok(target)["input_ids"]
        if not tgt_ids:
            print("  target tokenized to nothing, try again")
            continue
        tgt = tgt_ids[0]
        if len(tgt_ids) > 1:
            print(f"  (target is {len(tgt_ids)} tokens; using the first: {tok.decode([tgt])!r})")

        ids = tok(text, return_tensors="pt")["input_ids"]

        # ---- forward + loss + backward ----
        model.zero_grad(set_to_none=True)
        logits = model(ids).logits[0, -1]
        probs = torch.softmax(logits, -1)
        p_before = probs[tgt].item()
        rank = int((probs > probs[tgt]).sum().item()) + 1
        top = torch.topk(probs, 3)
        cands = "  ".join(f"{tok.decode([i])!r} {v*100:.1f}%" for v, i in zip(top.values, top.indices))
        print(f"\n  now:  p({tok.decode([tgt])!r}) = {p_before*100:.2f}%  (rank {rank} of 151,936)")
        print(f"  top-3: {cands}")

        loss = -torch.log(probs[tgt] + 1e-12)
        print(f"  loss = -ln p = {loss.item():.3f}   ...backpropagating...")
        loss.backward()

        # ---- credit assignment: gradient traffic per station ----
        traffic = []
        for lyr in model.model.layers:
            g = 0.0
            for proj in (lyr.self_attn.q_proj, lyr.self_attn.k_proj,
                         lyr.self_attn.v_proj, lyr.self_attn.o_proj):
                if proj.weight.grad is not None:
                    g += proj.weight.grad.norm().item() ** 2
            traffic.append(g ** 0.5)
        print("\n  gradient traffic through each station's attention (q/k/v/o):")
        bb = bars(traffic)
        for i in (0, 5, 11, 17, 23):
            print(f"    station {i+1:2d} {bb[i]:<31s} {traffic[i]:.4f}")
        top_st = max(range(24), key=lambda i: traffic[i])
        print(f"    (all 24 computed; loudest = station {top_st+1})")

        # per-head split of the loudest station's question-lens gradient
        qg = model.model.layers[top_st].self_attn.q_proj.weight.grad
        head_g = [qg[h*64:(h+1)*64, :].norm().item() for h in range(14)]
        hb = bars(head_g, width=20)
        print(f"\n  station {top_st+1}, question-lens nudge per head:")
        for h in range(14):
            print(f"    h{h:<2d} {hb[h]:<21s} {head_g[h]:.2e}")

        # which dictionary rows got blamed hardest
        eg = model.model.embed_tokens.weight.grad
        if eg is not None:
            row_norms = eg.norm(dim=1)
            tops = torch.topk(row_norms, 3)
            rows = "  ".join(f"{tok.decode([i])!r}" for i in tops.indices)
            print(f"\n  dictionary rows nudged hardest: {rows}")

        # ---- apply ONE nudge (clipped, as real training does) ----
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= args.lr * p.grad
        model.zero_grad(set_to_none=True)

        with torch.no_grad():
            probs2 = torch.softmax(model(ids).logits[0, -1], -1)
        p_after = probs2[tgt].item()
        rank2 = int((probs2 > probs2[tgt]).sum().item()) + 1
        print(f"\n  ONE NUDGE APPLIED (lr={args.lr}):")
        print(f"  p({tok.decode([tgt])!r}): {p_before*100:.2f}% -> {p_after*100:.2f}%   rank {rank} -> {rank2}")
        print(f"  (nudges accumulate -- repeat to keep training, 'reset' to undo everything)\n")


if __name__ == "__main__":
    main()
