"""Logit lens: watch the answer crystallize station by station.

At every station, take the last token's stream vector AS IT IS at that depth,
push it through the exit (final norm + dictionary scoring), and see which pages
it already points at. The model never does this mid-run -- we are peeking at
what the bet WOULD be if the machine stopped early.

Usage:  python lens.py          then type any text at TEXT>
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
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
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        final_probs = torch.softmax(out.logits[0, -1], -1)
        winner = final_probs.argmax()
        wtxt = tok.decode([winner])

        print(f"\n  final answer: {wtxt!r} ({final_probs[winner]*100:.1f}%)")
        print(f"  what the stream already pointed at, station by station:\n")
        print(f"  {'depth':>10}  {'top page (share)':<26} {'p(final winner)':>15}")
        last = len(out.hidden_states) - 1
        for i, hs in enumerate(out.hidden_states):
            with torch.no_grad():
                # the final entry is already exit-rescaled by the model
                v = hs[0, -1] if i == last else model.model.norm(hs[0, -1])
                logits = model.lm_head(v)                # score all 151,936 pages
                p = torch.softmax(logits, -1)
            top = p.argmax()
            name = "page" if i == 0 else f"station {i}"
            mark = "  <- winner appears" if top == winner and i < 24 else ""
            print(f"  {name:>10}  {tok.decode([top])!r:<20} {p[top]*100:5.1f}%  {p[winner]*100:14.2f}%{mark}")
        print()


if __name__ == "__main__":
    main()
