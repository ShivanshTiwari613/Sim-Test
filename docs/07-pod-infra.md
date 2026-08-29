# 07 · podenv/ — the GPU runs, and the five failures

Training needs a CUDA GPU; the Mac doesn't have one. `podenv/pod.py` is a
self-contained RunPod lifecycle CLI (stdlib-only locally — no pip installs on
the Mac), adapted from the sagaforge `forge.py` pattern but fully independent:
own pod name (`simtest-worldmodel`), own state file, own dedicated SSH key.

## The one command that matters

```
python3 podenv/pod.py run
  └► ensure pod exists & running (create on 4090→3090→A5000→A40 priority)
  └► wait for SSH (public IP + mapped port 22, dedicated ed25519 key)
  └► rsync code + data → /workspace/sim-test
  └► bootstrap: GPU preflight → venv on the persistent volume → pinned deps
  └► python train.py && python eval.py     (streamed to your terminal)
  └► rsync back checkpoints/ + eval_drift.png
  └► STOP the pod — in a finally, even on failure or Ctrl-C
```

Key economics: the 30GB `/workspace` volume survives stop/start and holds the
venv + HuggingFace cache, so only the first run pays for downloads. A stopped
pod bills ~pennies for storage; `destroy` ends everything. The full successful
run cost ≈ **$0.15**.

## The five failures (each now handled automatically)

1. **A dead key shadowed the live one.** A stale `RUNPOD_API_KEY` exported in
   `~/.zshrc` 401'd everything while the fresh key sat in `.env`.
   *Fix:* project `.env` now outranks the inherited environment.
2. **No rsync on the pod image** — and the container disk resets on every
   stop, so a one-time install can't stick.
   *Fix:* idempotent `apt-get install rsync` before every push.
3. **Latest transformers refused torch 2.4.** Unpinned pip pulled a build
   assuming torch ≥ 2.6; it died at import (`NameError: name 'torch'`).
   *Fix:* pinned transformers 4.46.3 / peft 0.14.0 / accelerate 1.2.1.
4. **A rented GPU that couldn't compute.** `nvidia-smi` looked healthy but raw
   `cuInit → 999` (broken host driver); torch silently fell back to CPU at
   446 s/step — an 800-hour ETA. Stopped pods restart on the *same* host, so
   stopping can't fix it.
   *Fix:* bootstrap runs a cuInit preflight first and exits 42; pod.py sees
   42 and **terminates** (not stops) so the rerun lands on new hardware.
5. **22% GPU utilization.** Batch 8 on ~120-token sequences starves a 4090 —
   steps were kernel-launch overhead, not math.
   *Fix:* batch 64, no grad accum: ~14 → ~190 examples/sec; the hour became
   10 minutes.

The meta-lesson: none of these were ML problems. The distance between "the
code works" and "the run works" is environment, versions, hardware, and
utilization — and each fix got encoded into the tooling so it can't happen
twice.

## Cheat sheet

```
python3 podenv/pod.py status      # state, $/hr, ssh endpoint, reachable?
python3 podenv/pod.py up / down   # start / stop (volume persists)
python3 podenv/pod.py shell       # interactive SSH
python3 podenv/pod.py pull        # fetch checkpoints + plots
python3 podenv/pod.py destroy     # terminate entirely (y/N confirm)
python3 podenv/pod.py --dry-run run   # print every request, no key needed
```
