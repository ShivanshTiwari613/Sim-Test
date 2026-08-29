# Toy Text World Model

The smallest useful world model: **Qwen2.5-0.5B fine-tuned with LoRA to predict
a gridworld's next state from text alone.** The model takes `(state, action)` as
a string and emits `state'` as a string. The state string *is* the memory —
nothing lives outside the prompt. Built to learn the mechanics of training a
world model, not to be a production component.

## Results (2026-08-25 run)

| Metric | Value |
|---|---|
| Single-step exact match (500 held-out transitions) | **98.0%** |
| "State unchanged" trivial baseline | 42.6% |
| Rollout exact-match at step 10 (model's own predictions fed back) | **96.0%** |
| Episodes exactly correct through all 10 steps | 48 / 50 |
| Final train loss (3 epochs) | ~0.003 |
| Training wall-clock (RTX 4090, batch 64) | 10 min |
| Total pod cost for the run | ≈ $0.15 |

The headline finding is the **flat drift curve** (`eval_drift.png`): 98% at
step 1 → 96% at step 10. Errors don't compound — a correctly predicted state is
a fully self-contained input for the next step, so the 48 clean episodes stay
exact forever; the 2 that diverge do so once and stay diverged.

### Autopsy (2026-08-29, full test set — `autopsy.py`)

Re-run over **all** 3,000 held-out transitions and all 150 rollout episodes:
**99.63%** single-step (2,989/3,000), 148/150 rollouts clean. Every one of the
13 misses is the **same failure mode**: a phantom move through a wall — the
world said no-op, the model moved the agent anyway. Only 3 unique
(state, action) pairs are involved (one stuck episode repeats its miss 10×).
The correlation: wall-blocks are the *rarest* blocking cause in training
(1,230 examples vs 2,405 edge-blocks), south-wall-blocks the rarest direction
(226), and unlike edges (constant geometry) or the door (textually flagged
`locked`), a wall-block requires coordinate math against the walls list.
The model's only weakness is exactly its least-taught, hardest-to-infer rule.
Details: `autopsy_report.txt`, `autopsy_misses.jsonl`, `logs/autopsy.log`.

## The environment

Deterministic 5×5 grid (`env.py`): agent, key, locked door, 0–2 walls.
Actions: `move north/south/east/west`, `pick up`, `unlock`. (0,0) is
south-west, north = y+1. Invalid actions are no-ops. `unlock` needs the key in
hand and orthogonal adjacency to the door; an unlocked door cell is passable.

Canonical serialization — fixed ordering, spacing, sorted walls — so model
capacity goes to dynamics, not formatting noise:

```
agent=(2,3); key=(0,1); door=(4,4) locked; has_key=false; walls=[(1,1),(3,2)]
```

`key=held` + `has_key=true` after pick-up is deliberately redundant: eval
catches a model that lets the two fields drift apart.

## Pipeline

```bash
pip install -r requirements.txt      # only needed to train/eval locally

python env.py         # demo the gridworld + serializer (stdlib only)
python gen_data.py    # -> data/train.jsonl (35k) + data/test.jsonl (3k)
python train.py       # LoRA SFT; --smoke for a 200-example sanity run
python eval.py        # single-step + rollout drift -> eval_drift.png
```

**Data** (`gen_data.py`): 1,750 train / 150 test episodes of 20 steps, split
**by episode** so multi-step eval never sees a training episode. The rollout
policy is lightly goal-biased (drift toward key, then door, then interact) so
rare transitions are represented: ~1.3k successful pick-ups, ~0.8k successful
unlocks. ~57% of train transitions change the state — hence the 42.6% baseline.

**Training** (`train.py`): LoRA r=16 on q/k/v/o projections, lr 2e-4,
3 epochs, batch 64 (drop to 8 + `--grad-accum 2` on a T4/MPS). Loss is masked
to the completion only. Prompt template:

```
STATE: <state>
ACTION: <action>
NEXT: <next_state>
```

**Eval** (`eval.py`): exact-match on held-out transitions vs. the unchanged
baseline, plus a 10-step rollout where the model's own predictions are fed
back with the episode's true action sequence.

## Running on a RunPod GPU (`podenv/`)

Self-contained pod lifecycle (stdlib-only on the local machine), adapted from
the sagaforge `forge.py` pattern but fully independent — own pod name, state
file, and SSH key:

```bash
# key from $RUNPOD_API_KEY or the RUNPOD_API_KEY= line in .env
python3 podenv/pod.py run        # up -> push -> deps -> train+eval -> pull
                                 # checkpoints + plot -> STOP pod (always)
python3 podenv/pod.py status     # state, $/hr, ssh endpoint
python3 podenv/pod.py down       # stop (volume persists)
python3 podenv/pod.py destroy    # terminate (volume gone, cost -> $0)
python3 podenv/pod.py --dry-run run   # print every request, no key needed
```

Spec: `runpod/pytorch` CUDA image, GPU priority 4090 → 3090 → A5000 → A40,
30GB `/workspace` volume holding the venv + HF cache so restarts skip all
downloads. `run` stops the pod in a `finally` even on Ctrl-C.

### Field notes from the first full run

Every one of these is now handled automatically — kept here because they're
the actual lessons of the exercise:

1. **Stale API key shadowing the real one.** A dead `RUNPOD_API_KEY` exported
   in `~/.zshrc` produced blanket 401s while the fresh key sat in `.env`.
   Fix: `pod.py` now prefers the project `.env` over the inherited env.
2. **No rsync on the pod image.** `runpod/pytorch` doesn't ship it, and the
   container disk resets on every stop. Fix: idempotent `apt-get install
   rsync` before each push.
3. **Latest transformers vs. the image's torch 2.4.** New transformers
   releases assume torch ≥ 2.6 and die at import (`NameError: torch`). Fix:
   pinned `transformers==4.46.3`, `peft==0.14.0`, `accelerate==1.2.1`.
4. **A broken GPU host.** `nvidia-smi` looked fine but raw `cuInit → 999`;
   torch silently fell back to CPU at 446 s/step (~800 hours). A stopped pod
   restarts on the same host, so stopping doesn't help. Fix: `bootstrap.sh`
   runs a cuInit preflight first and exits 42; `pod.py` sees 42 and
   **terminates** the pod so the next run lands on a different machine.
5. **22% GPU utilization.** Batch 8 on ~120-token sequences starves a 4090 —
   steps were all kernel-launch overhead. Batch 64 (no grad accum) took
   throughput from ~14 to ~190 examples/sec: 10-minute training instead of an
   hour.

## Repo layout

```
env.py            gridworld + canonical serializer (stdlib)
gen_data.py       rollouts -> data/train.jsonl, data/test.jsonl
train.py          LoRA SFT (CUDA / MPS / CPU)
eval.py           single-step + rollout drift -> eval_drift.png
podenv/           independent RunPod lifecycle (pod.py, bootstrap.sh)
docs/             study notes: env → data → tokens → LoRA → eval, from zero
data/             35k train / 3k test transitions (episode-split)
checkpoints/      trained LoRA adapter (~8.7MB)
eval_drift.png    the drift curve
```

## Provenance & attestation

Every number in this README (and any post citing it) can be checked from this
repo alone. Exact provenance of the published run:

| What | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B` (Hugging Face) |
| Fine-tune | LoRA r=16 on q/k/v/o projections, lr 2e-4, 3 epochs, batch 64 |
| Data seed | `42` (single seeded `random.Random`; regeneration is byte-identical) |
| Pinned deps | `transformers==4.46.3`, `peft==0.14.0`, `accelerate==1.2.1` (see `requirements.txt`) |
| Training run | 2026-08-25, RunPod RTX 4090, ~10 min, ≈ $0.15 |
| Full-test autopsy | 2026-08-29, all 3,000 test transitions + 150 rollouts |

**What each committed file attests:**

- `data/train.jsonl` / `data/test.jsonl` — the exact 35,000 / 3,000 transitions
  the model was trained and evaluated on. Verify: `python gen_data.py` into a
  temp dir and `diff` — seed 42 reproduces them byte-for-byte.
- `checkpoints/qwen05b-lora/` — the trained adapter itself
  (`adapter_model.safetensors`, ~8.3 MB, 2,162,688 trainable params = 0.44% of
  the base model). Verify the headline claim without training anything:
  `python eval.py` loads this adapter and re-derives the 98.0% / 96.0% numbers.
- `eval_drift.png` — the drift curve as produced by that eval run.
- `autopsy_report.txt`, `autopsy_misses.jsonl`, `logs/autopsy.log` — the full
  miss list behind the 99.63% autopsy: all 13 misses, raw, including the
  repeated ones. Nothing was filtered.
- `env.py` / `gen_data.py` / `train.py` / `eval.py` / `autopsy.py` — the entire
  pipeline; there is no step that isn't in the repo.

**Deliberately excluded** (see `.gitignore`): `.env` (RunPod API key),
`podenv/id_ed25519*` and `podenv/.pod_state.json` (SSH key + machine state for
my pod — the lifecycle *script* is committed, the credentials are not),
intermediate training checkpoints (~75 MB of optimizer state; the final
adapter is what the claims rest on), and `report/` (personal study notes).

Three levels of verification, cheapest first:

1. **Check the numbers** — run `eval.py` against the committed adapter and
   data. CPU/MPS works; no GPU needed.
2. **Check the data** — regenerate with seed 42 and diff against the committed
   JSONL.
3. **Check everything** — retrain from scratch (`train.py`, or
   `podenv/pod.py run` on a rented GPU for ~$0.15) and reproduce the adapter.

## Next experiments

- Inspect the 2 diverging episodes — which rule do they break?
- Stochastic env (slip probability) — exact match stops being the right metric.
- The nanoGPT-style ~1M-param from-scratch track on the same JSONL.
- Reading: Ha & Schmidhuber "World Models" (2018), then Dreamer v1–v3 — the
  same loop with a learned latent instead of a text string.
