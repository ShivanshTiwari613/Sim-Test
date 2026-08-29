---
base_model: Qwen/Qwen2.5-0.5B
library_name: peft
pipeline_tag: text-generation
tags:
  - lora
  - world-model
  - gridworld
---

# qwen05b-lora — toy text world model adapter

A LoRA adapter that turns `Qwen/Qwen2.5-0.5B` into a next-state predictor for
a deterministic 5×5 gridworld (agent, key, locked door, 0–2 walls). Input is
a state string plus an action; output is the exact next state string. The
full project — environment, data generator, training and eval code, and every
measured number — lives one directory up in this repository.

## How to load

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
model = PeftModel.from_pretrained(base, "checkpoints/qwen05b-lora")
tok = AutoTokenizer.from_pretrained("checkpoints/qwen05b-lora")

prompt = ("STATE: agent=(2,3); key=(0,1); door=(4,4) locked; has_key=false; "
          "walls=[(1,1),(3,2)]\nACTION: move south\nNEXT: ")
out = model.generate(**tok(prompt, return_tensors="pt"),
                     max_new_tokens=80, do_sample=False)
print(tok.decode(out[0][tok(prompt, return_tensors="pt").input_ids.shape[1]:],
                 skip_special_tokens=True))
# -> agent=(2,2); key=(0,1); door=(4,4) locked; has_key=false; walls=[(1,1),(3,2)]
```

Use greedy decoding (`do_sample=False`); the world is deterministic.

## Adapter details

- LoRA r=16, alpha per `adapter_config.json`, on q/k/v/o projections of all
  24 attention layers: 192 tensors, 2,162,688 trainable parameters (0.44% of
  the 494M-weight base). Adapter file: 8.7 MB.
- The `checkpoint-547/1094/1641` subfolders (per-epoch save points with
  optimizer state) are not committed — only the final adapter is.

## Training

Supervised fine-tuning on 35,000 (state, action → next state) transitions
from `../../data/train.jsonl` (seed 42, split by episode). lr 2e-4, 3 epochs,
batch 64, loss masked to the completion (prompt labels −100). ~10 minutes on
a rented RTX 4090; final train loss ≈ 0.003. Recipe: `../../train.py`.

## Evaluation

On 3,000 held-out transitions from never-seen episodes, graded by exact
string match: **99.63%** single-step; **148/150** episodes exact through
10-step rollouts feeding the model its own predictions back. Trivial
"predict no change" baseline: 42.6%. Full tables: `../../METRICS.md`.

## Limitations

- All 13 known mispredictions are one failure mode: on a move blocked by a
  wall, the model sometimes moves the agent anyway (90.18% on that rule,
  100% on the other seven). Raw misses: `../../autopsy_misses.jsonl`.
- The adapter only knows this gridworld's canonical state format; it is not
  a general model and inherits Qwen2.5-0.5B's license and behaviors for
  anything outside it.

### Framework versions

- PEFT 0.14.0 · transformers 4.46.3 · accelerate 1.2.1 (pinned in
  `../../requirements.txt`)
