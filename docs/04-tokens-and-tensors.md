# 04 · Tokens and tensors — what the model actually receives

The model never sees letters. It sees **rows of integers**. This doc uses the
real first training row (episode 0, step 0: agent moves north from (3,3) to
(3,4)), tokenized with the actual Qwen tokenizer from the trained checkpoint.

## Token ids are dictionary page numbers

Qwen's tokenizer owns a fixed dictionary of **151,936** text pieces, each with
a row number. Tokenizing = looking pieces up:

| piece | id | | piece | id |
|---|---|---|---|---|
| `STATE` | 24628 | | `3` | 18 |
| `=(` | 4539 | | `4` | 19 |
| `);` | 1215 | | ` locked` | 16061 |
| ` walls` | 14285 | | `),(` | 23547 |
| `<|endoftext|>` | 151643 | | ` ` (space) | 220 |

**The size of an id means nothing** — 16061 is not "big", it's just where
" locked" happens to live. Two lucky facts: digits 0–4 are single tokens
(so coordinates never get sliced mid-number), and punctuation clumps into
stable units (`=(`, `);`, `),(`), so the state format tokenizes identically
every time.

## One training example = three aligned integer rows

The example: prompt = 45 tokens, completion = 36 tokens (35 state tokens +
eos), total 81.

```
position:        0      1      2    …    44  │  45     46   …   49   …   80
text:          STATE    :    ␣agent  …   ␣   │ agent   =(   …    4   …  <eos>
input_ids:     24628   25    8315   …   220  │ 8092   4539  …   19   …  151643
labels:        -100   -100   -100   …  -100  │ 8092   4539  …   19   …  151643
attention:       1      1      1    …    1   │   1      1   …    1   …    1
                └────── read, never graded ──┘ └──── every one graded ───────┘
```

- **input_ids** — what the model reads (the whole sequence).
- **labels** — what it's graded on. `-100` is PyTorch's "skip this position"
  sentinel: all 45 prompt positions are masked, so no gradient flows from
  reproducing the question. Only the answer earns loss. (This is what
  "loss on the completion only" means.)
- **attention_mask** — 1 = look at this position; 0 appears only over batch
  padding.

## Where the physics lives: one token

Diff the graded half against the prompt:

```
prompt   pos  6:  id 18  "3"   ← agent's y before
compl.   pos 49:  id 19  "4"   ← agent's y after     ★ the ONLY changed token
```

35 of 36 graded tokens are **copy-work** — key, door, has_key, and all 11
wall tokens appear in both halves and must be reproduced exactly (walls never
move; getting them wrong fails exact-match without any physics
misunderstanding). One token carries the rule: emit 19 where the input had 18.
A pick-up example changes ~4 tokens (`=(3,4)` → `=held`, `=false` → `=true`);
a no-op example changes zero.

The last graded token is **151643** `<|endoftext|>` — the stop sign is
trained like any other token, which is why generation halts cleanly at eval
time instead of inventing more world.

## Batching: × 64

The collator stacks 64 examples into rectangles, padding shorter ones up to
the batch's longest (~85 tokens, not the 512 cap — padding per-batch wastes
4× less compute):

```
input_ids:       64 × ~85 integers      (pad token in the gaps)
labels:          64 × ~85               (-100 in the gaps — filler never graded)
attention_mask:  64 × ~85 of 1/0        (0 over filler — never even looked at)
```

One forward pass grades ~2,300 positions; one backward pass nudges the 2.16M
LoRA weights; that is one of the run's 1,641 steps. Total: ≈ 3.7 million
graded token predictions to learn a rulebook with ~1 changed token per row.

*Interactive version: the Number Atlas artifact has all 81 tokens clickable.*
