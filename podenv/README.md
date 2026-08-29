# podenv — self-contained RunPod training environment

Independent pod lifecycle for this project (pattern borrowed from
`sagaforge/scripts/forge.py`, but fully separate: own pod name
`simtest-worldmodel`, own state file, own SSH key, and it never touches the
sagaforge pod). Stdlib-only — needs no pip packages on the local machine,
just `ssh`/`rsync`/`ssh-keygen` (all present on macOS).

## Usage

```bash
# key comes from $RUNPOD_API_KEY or the RUNPOD_API_KEY= line in ../.env
python3 podenv/pod.py run            # up -> push code+data -> install deps ->
                                     # train.py && eval.py -> pull checkpoints
                                     # + eval_drift.png -> STOP pod (always)
python3 podenv/pod.py run train      # just training
python3 podenv/pod.py run eval       # just eval (needs a trained adapter)
python3 podenv/pod.py run --keep-up  # leave the pod running afterwards

python3 podenv/pod.py up             # start pod, wait for SSH
python3 podenv/pod.py status         # state, $/hr, SSH endpoint, reachability
python3 podenv/pod.py push           # rsync code + data/ to /workspace/sim-test
python3 podenv/pod.py pull           # fetch checkpoints/ + eval outputs
python3 podenv/pod.py shell          # interactive SSH
python3 podenv/pod.py down           # stop (billing stops, volume persists)
python3 podenv/pod.py destroy        # terminate entirely (y/N confirm)

python3 podenv/pod.py --dry-run run  # print every request/command, no key needed
```

## What `up` creates

- `runpod/pytorch:2.4.0-py3.11-cuda12.4.1` container, 1 GPU from priority
  list: RTX 4090 → RTX 3090 → RTX A5000 → A40 (community cloud first,
  secure-cloud fallback)
- 30GB volume at `/workspace` holding the venv + HuggingFace cache
  (`HF_HOME=/workspace/hf`), so stop/start never re-downloads Qwen or
  reinstalls packages — only `destroy` loses them
- TCP port 22 exposed; access via a dedicated ed25519 key generated into
  `podenv/id_ed25519` (gitignored) and injected through the image's
  `PUBLIC_KEY` hook

State lives in `podenv/.pod_state.json` (pod id). A saved id that 404s is
forgotten and a fresh pod is created; a pod stranded on a full host machine is
terminated and recreated automatically. `run` stops the pod in a `finally`
even on failure or Ctrl-C — if you ever see a stop warning, check the RunPod
console.
