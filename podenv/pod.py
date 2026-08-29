#!/usr/bin/env python3
"""Self-contained RunPod training-pod lifecycle for the Sim-Test world model.

Independent of any other project: own pod name, own state file, own SSH key,
stdlib-only (no pip installs needed on this machine). Driven entirely through
subcommand switches:

  pod.py up                 ensure the "simtest-worldmodel" pod exists and is
                            running, wait for SSH to answer
  pod.py status             pod state, cost/hr, SSH endpoint, reachability
  pod.py push               rsync the project (code + data/) to /workspace/sim-test
  pod.py run [train|eval|all]   up -> push -> bootstrap deps -> run the job on
                            the pod -> pull checkpoints/ and eval outputs back
                            -> stop the pod, ALWAYS (even on failure/Ctrl-C)
  pod.py pull               fetch checkpoints/ + eval outputs from the pod
  pod.py shell              interactive SSH into the pod
  pod.py down               stop the pod (billing stops; volume persists)
  pod.py destroy            terminate the pod entirely (y/N confirm)

  --dry-run                 print the HTTP requests / shell commands instead of
                            executing them; works without RUNPOD_API_KEY
  --keep-up                 (run) leave the pod running afterwards

Auth: RUNPOD_API_KEY from the environment, or from .env / podenv/.env at the
repo root (KEY=value lines).

Pod spec: runpod/pytorch CUDA image, 1x GPU from a priority list (4090 ->
3090 -> A5000 -> A40, all 24GB+), 30GB volume at /workspace. The volume keeps
a venv and the HF model cache, so only the first run pays for downloads.
SSH uses a dedicated ed25519 key generated into podenv/ (gitignored) and
injected via the image's PUBLIC_KEY env hook; the connection goes to the
pod's public IP on the mapped TCP port for 22.

Uses the RunPod REST API (https://rest.runpod.io/v1): POST/GET /pods,
GET /pods/{id}, POST /pods/{id}/start|stop, DELETE /pods/{id}, auth via
`Authorization: Bearer <RUNPOD_API_KEY>`.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PODENV = Path(__file__).resolve().parent
ROOT = PODENV.parent
STATE_FILE = PODENV / ".pod_state.json"
SSH_KEY = PODENV / "id_ed25519"

API_BASE = "https://rest.runpod.io/v1"

# --- pod spec (edit here to change what `up` creates) ---
POD_NAME = "simtest-worldmodel"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
GPU_TYPE_IDS = [
    # priority order — RunPod rents the first type with capacity. All 24GB+,
    # ample for 0.5B + LoRA; 3090/A5000 are often cheaper than the 4090.
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A5000",
    "NVIDIA A40",
]
CONTAINER_DISK_GB = 20
VOLUME_GB = 30
VOLUME_MOUNT_PATH = "/workspace"
REMOTE_DIR = "/workspace/sim-test"
REMOTE_VENV = "/workspace/venv"
REMOTE_HF_HOME = "/workspace/hf"

WAIT_TIMEOUT_S = 600
POLL_INTERVAL_S = 5
DRYRUN_POD_ID = "dryrun-pod-id"

SYNC_INCLUDES = ["env.py", "gen_data.py", "train.py", "eval.py", "autopsy.py",
                 "dream.py", "requirements.txt", "data", "podenv/bootstrap.sh"]
PULL_PATHS = ["checkpoints", "eval_drift.png", "autopsy_misses.jsonl",
              "autopsy_report.txt"]


# --- local state ---

def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")


def clear_state():
    if STATE_FILE.exists():
        STATE_FILE.unlink()


# --- auth ---

def load_dotenv_key():
    for env_file in (ROOT / ".env", PODENV / ".env"):
        if not env_file.exists():
            continue
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("RUNPOD_API_KEY="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val:
                    return val
    return None


def require_api_key(dry_run):
    if dry_run:
        return None
    # Project .env wins over the inherited environment so a stale global
    # export (e.g. in ~/.zshrc) can't shadow this project's key.
    key = load_dotenv_key() or os.environ.get("RUNPOD_API_KEY")
    if not key:
        print("error: RUNPOD_API_KEY not set (env or .env). Get one at "
              "runpod.io -> Settings -> API Keys.", file=sys.stderr)
        sys.exit(1)
    return key


# --- SSH key (dedicated to this project) ---

def ensure_ssh_key(dry_run):
    if SSH_KEY.exists():
        return SSH_KEY.with_suffix(".pub").read_text().strip()
    if dry_run:
        print(f"[dry-run] would generate ssh key at {SSH_KEY}")
        return "ssh-ed25519 DRYRUN simtest-worldmodel"
    print(f"generating dedicated ssh key: {SSH_KEY}")
    subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-N", "", "-C", "simtest-worldmodel",
         "-f", str(SSH_KEY)],
        check=True, stdout=subprocess.DEVNULL,
    )
    return SSH_KEY.with_suffix(".pub").read_text().strip()


# --- RunPod REST API (stdlib urllib; no dependencies) ---

def api_request(method, path, api_key, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{API_BASE}{path}", data=data, method=method,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode()
            return resp.status, (json.loads(text) if text else None)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def fail(status, path, payload):
    print(f"error: {path} -> HTTP {status}", file=sys.stderr)
    print(payload, file=sys.stderr)
    sys.exit(1)


def build_pod_spec(cloud_type, public_key):
    return {
        "name": POD_NAME,
        "imageName": IMAGE,
        "gpuTypeIds": GPU_TYPE_IDS,
        "gpuCount": 1,
        "cloudType": cloud_type,
        "containerDiskInGb": CONTAINER_DISK_GB,
        "volumeInGb": VOLUME_GB,
        "volumeMountPath": VOLUME_MOUNT_PATH,
        "ports": ["22/tcp"],
        "env": {"PUBLIC_KEY": public_key},
    }


def create_pod(api_key, public_key, dry_run):
    """POST /pods; COMMUNITY first (cheaper), SECURE fallback on no capacity."""
    for cloud_type in ("COMMUNITY", "SECURE"):
        body = build_pod_spec(cloud_type, public_key)
        if dry_run:
            print(f"[dry-run] POST {API_BASE}/pods (cloudType={cloud_type})")
            print(json.dumps(body, indent=2))
            return {"id": DRYRUN_POD_ID}
        status, payload = api_request("POST", "/pods", api_key, body)
        if status in (200, 201):
            return payload
        print(f"POST /pods (cloudType={cloud_type}) -> {status}: {payload}",
              file=sys.stderr)
        if cloud_type == "COMMUNITY":
            print("retrying with cloudType=SECURE...")
            continue
        sys.exit(1)


def get_pod(pod_id, api_key, dry_run):
    if dry_run:
        print(f"[dry-run] GET {API_BASE}/pods/{pod_id}")
        return None
    status, payload = api_request("GET", f"/pods/{pod_id}", api_key)
    if status == 404:
        return None
    if status != 200:
        fail(status, f"GET /pods/{pod_id}", payload)
    return payload


def start_pod(pod_id, api_key, dry_run):
    if dry_run:
        print(f"[dry-run] POST {API_BASE}/pods/{pod_id}/start")
        return
    status, payload = api_request("POST", f"/pods/{pod_id}/start", api_key)
    if status != 200:
        raise RuntimeError(f"start -> HTTP {status}: {payload}")


def stop_pod(pod_id, api_key, dry_run):
    if dry_run:
        print(f"[dry-run] POST {API_BASE}/pods/{pod_id}/stop")
        return
    status, payload = api_request("POST", f"/pods/{pod_id}/stop", api_key)
    if status != 200:
        fail(status, f"POST /pods/{pod_id}/stop", payload)


def terminate_pod(pod_id, api_key, dry_run):
    if dry_run:
        print(f"[dry-run] DELETE {API_BASE}/pods/{pod_id}")
        return
    status, payload = api_request("DELETE", f"/pods/{pod_id}", api_key)
    if status not in (200, 204):
        fail(status, f"DELETE /pods/{pod_id}", payload)


def ensure_pod(api_key, dry_run):
    """Create the pod if no saved id (or the saved one 404s), then start it
    if it isn't running. Self-heals a pod stranded on a full host machine."""
    public_key = ensure_ssh_key(dry_run)
    state = load_state()
    pod_id = state.get("pod_id")
    pod = None
    if pod_id:
        pod = get_pod(pod_id, api_key, dry_run)
        if not dry_run and pod is None:
            print(f"saved pod {pod_id} no longer exists; creating a new one")
            pod_id = None

    if pod_id is None:
        pod = create_pod(api_key, public_key, dry_run)
        pod_id = pod["id"]
        if not dry_run:
            save_state({"pod_id": pod_id})
        print(f"pod: {pod_id} (created)")
    else:
        print(f"pod: {pod_id} (existing)")

    desired = (pod or {}).get("desiredStatus")
    if dry_run:
        print(f"[dry-run] POST {API_BASE}/pods/{pod_id}/start")
    elif desired == "RUNNING":
        print(f"pod {pod_id} already running")
    else:
        print(f"starting pod {pod_id} (status was {desired})...")
        try:
            start_pod(pod_id, api_key, dry_run)
        except RuntimeError as e:
            # A stopped pod is pinned to its host machine; if that host's
            # GPUs are taken, start fails forever. Terminate and recreate
            # elsewhere (costs a re-download — the volume dies with the pod).
            if "not enough free GPUs" in str(e) or "resources to deploy" in str(e):
                print(f"host for pod {pod_id} is full — recreating on a new host")
                terminate_pod(pod_id, api_key, dry_run)
                clear_state()
                pod = create_pod(api_key, public_key, dry_run)
                pod_id = pod["id"]
                save_state({"pod_id": pod_id})
                print(f"pod: {pod_id} (recreated)")
            else:
                print(f"error: {e}", file=sys.stderr)
                sys.exit(1)

    return pod_id


# --- SSH plumbing ---

def ssh_endpoint(pod, dry_run):
    """(host, port) for the pod's mapped TCP port 22, or None until ready."""
    if dry_run:
        return ("dryrun-host", 22)
    ip = pod.get("publicIp") or ""
    mappings = pod.get("portMappings") or {}
    port = mappings.get("22")
    if ip and port:
        return (ip, int(port))
    return None


def ssh_base_args(port):
    return ["-i", str(SSH_KEY), "-p", str(port),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
            "-o", "ConnectTimeout=10"]


def wait_for_ssh(pod_id, api_key, dry_run, timeout=WAIT_TIMEOUT_S):
    if dry_run:
        print(f"[dry-run] would poll GET /pods/{pod_id} for publicIp + port "
              f"22 mapping, then `ssh true`, up to {timeout}s")
        return ("dryrun-host", 22)
    print(f"waiting for SSH on pod {pod_id} (up to {timeout}s)...")
    deadline = time.time() + timeout
    endpoint = None
    while time.time() < deadline:
        pod = get_pod(pod_id, api_key, dry_run)
        if pod is not None:
            endpoint = ssh_endpoint(pod, dry_run)
            if endpoint:
                host, port = endpoint
                probe = subprocess.run(
                    ["ssh"] + ssh_base_args(port) + [f"root@{host}", "true"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                if probe.returncode == 0:
                    print(f"ssh is up: root@{host}:{port}")
                    return endpoint
        time.sleep(POLL_INTERVAL_S)
    print(f"error: SSH on pod {pod_id} not reachable within {timeout}s",
          file=sys.stderr)
    sys.exit(1)


def gpu_preflight(endpoint, pod_id, api_key, dry_run):
    """Check cuInit over ssh the moment ssh works — BEFORE any slow push.
    On a broken host, terminate the pod (a stopped pod would restart on the
    same machine) and exit so the user can simply run `up` again."""
    if dry_run:
        print("[dry-run] would ssh a cuInit(0) probe")
        return
    host, port = endpoint
    probe = subprocess.run(
        ["ssh"] + ssh_base_args(port) + [f"root@{host}",
         "python3 -c \"import ctypes;import sys;"
         "sys.exit(ctypes.CDLL('libcuda.so.1').cuInit(0))\""],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if probe.returncode == 0:
        print("GPU preflight ok (cuInit 0)")
        return
    print(f"BROKEN GPU on this host (cuInit -> {probe.returncode}) — "
          "terminating pod; run `up` again to get a fresh host", file=sys.stderr)
    terminate_pod(pod_id, api_key, dry_run)
    clear_state()
    sys.exit(1)


def remote_exec(endpoint, command, dry_run, stream=True):
    host, port = endpoint
    cmd = ["ssh"] + ssh_base_args(port) + [f"root@{host}", command]
    if dry_run:
        print(f"[dry-run] {' '.join(cmd)}")
        return 0
    if stream:
        return subprocess.call(cmd)
    return subprocess.run(cmd, capture_output=True, text=True)


def rsync(src_list, dest, port, dry_run, delete=False, excludes=()):
    cmd = ["rsync", "-az", "--progress"]
    if delete:
        cmd.append("--delete")
    for e in excludes:
        cmd += ["--exclude", e]
    cmd += ["-e", "ssh " + " ".join(ssh_base_args(port))]
    cmd += src_list + [dest]
    if dry_run:
        print(f"[dry-run] {' '.join(cmd)}")
        return 0
    return subprocess.call(cmd)


# --- subcommands ---

def cmd_up(args):
    api_key = require_api_key(args.dry_run)
    pod_id = ensure_pod(api_key, args.dry_run)
    host, port = wait_for_ssh(pod_id, api_key, args.dry_run)
    gpu_preflight((host, port), pod_id, api_key, args.dry_run)
    print(f"\npod up: {pod_id}")
    print(f"ssh: ssh -i {SSH_KEY} -p {port} root@{host}")
    print("run `pod.py down` when finished to stop billing.")


def cmd_status(args):
    if args.dry_run:
        print("[dry-run] would GET the saved pod (if any)")
        return
    api_key = require_api_key(False)
    pod_id = load_state().get("pod_id")
    if not pod_id:
        print("no saved pod id — nothing created yet (run `pod.py up`)")
        return
    pod = get_pod(pod_id, api_key, dry_run=False)
    if pod is None:
        print(f"saved pod {pod_id} no longer exists (stale state; `up` recreates)")
        return
    desired = pod.get("desiredStatus", "?")
    print(f"pod: {pod_id}  name={pod.get('name')}  status={desired}  "
          f"gpu={pod.get('machine', {}).get('gpuTypeId', pod.get('gpuTypeId', '?'))}  "
          f"cost/hr=${pod.get('costPerHr', '?')}")
    endpoint = ssh_endpoint(pod, dry_run=False)
    if desired != "RUNNING":
        print("ssh: n/a (pod not running)")
    elif not endpoint:
        print("ssh: port mapping not assigned yet")
    else:
        host, port = endpoint
        probe = subprocess.run(
            ["ssh"] + ssh_base_args(port) + [f"root@{host}", "true"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print(f"ssh: root@{host}:{port}  reachable={probe.returncode == 0}")


def push_project(endpoint, dry_run):
    host, port = endpoint
    # The runpod/pytorch image has no rsync, and the container disk resets on
    # every stop — ensure it (quickly, idempotently) before each push.
    rc = remote_exec(
        endpoint,
        "command -v rsync >/dev/null || "
        "(apt-get update -qq && apt-get install -qq -y rsync) >/dev/null; "
        f"mkdir -p {REMOTE_DIR}/podenv",
        dry_run,
    )
    if rc != 0:
        print("error: failed to prepare pod for push", file=sys.stderr)
        sys.exit(1)
    srcs = [str(ROOT / p) for p in SYNC_INCLUDES if (ROOT / p).exists() or dry_run]
    # data/ and podenv/bootstrap.sh keep their dir structure via -R-style paths:
    # simplest is two syncs — flat files to REMOTE_DIR, data/ as a dir.
    flat = [s for s in srcs if not s.endswith(("data", "bootstrap.sh"))]
    rc = rsync(flat, f"root@{host}:{REMOTE_DIR}/", port, dry_run)
    rc |= rsync([str(ROOT / "data")], f"root@{host}:{REMOTE_DIR}/", port,
                dry_run, delete=True)
    rc |= rsync([str(PODENV / "bootstrap.sh")],
                f"root@{host}:{REMOTE_DIR}/podenv/", port, dry_run)
    # Push the trained adapter too (pods can be recreated on host failure,
    # taking their volume with them — the local copy is the source of truth).
    # The heavy per-epoch snapshots stay local.
    if (ROOT / "checkpoints" / "qwen05b-lora").exists() or dry_run:
        rc |= rsync([str(ROOT / "checkpoints")], f"root@{host}:{REMOTE_DIR}/",
                    port, dry_run,
                    excludes=["checkpoint-*", "tokenizer*", "vocab.json",
                              "merges.txt", "added_tokens.json",
                              "special_tokens_map.json"])
    if rc != 0:
        print("error: rsync push failed", file=sys.stderr)
        sys.exit(1)
    print(f"pushed project to {REMOTE_DIR}")


def pull_artifacts(endpoint, dry_run):
    host, port = endpoint
    rc = 0
    for p in PULL_PATHS:
        rc |= rsync([f"root@{host}:{REMOTE_DIR}/{p}"], str(ROOT) + "/",
                    port, dry_run)
    if rc != 0:
        print("warning: some artifacts failed to pull (may not exist yet)",
              file=sys.stderr)
    else:
        print("pulled checkpoints/ and eval outputs")


def connected_endpoint(args, require_running=True):
    api_key = require_api_key(args.dry_run)
    pod_id = load_state().get("pod_id")
    if args.dry_run:
        return (DRYRUN_POD_ID, ("dryrun-host", 22), None)
    if not pod_id:
        print("no saved pod id (run `pod.py up` first)", file=sys.stderr)
        sys.exit(1)
    pod = get_pod(pod_id, api_key, dry_run=False)
    if pod is None or (require_running and pod.get("desiredStatus") != "RUNNING"):
        print("pod is not running (run `pod.py up`)", file=sys.stderr)
        sys.exit(1)
    endpoint = ssh_endpoint(pod, dry_run=False)
    if not endpoint:
        print("pod has no ssh port mapping yet; try again shortly", file=sys.stderr)
        sys.exit(1)
    return (pod_id, endpoint, api_key)


def cmd_push(args):
    _, endpoint, _ = connected_endpoint(args)
    push_project(endpoint, args.dry_run)


def cmd_pull(args):
    _, endpoint, _ = connected_endpoint(args)
    pull_artifacts(endpoint, args.dry_run)


def cmd_shell(args):
    _, (host, port), _ = connected_endpoint(args)
    if args.dry_run:
        print(f"[dry-run] ssh -i {SSH_KEY} -p {port} root@{host}")
        return
    os.execvp("ssh", ["ssh"] + ssh_base_args(port) + [f"root@{host}"])


JOBS = {
    "train": "python train.py",
    "eval": "python eval.py",
    "autopsy": "python autopsy.py",
    "all": "python train.py && python eval.py",
}


def cmd_run(args):
    api_key = require_api_key(args.dry_run)
    pod_id = ensure_pod(api_key, args.dry_run)
    exit_code = 1
    terminated = False
    try:
        endpoint = wait_for_ssh(pod_id, api_key, args.dry_run)
        gpu_preflight(endpoint, pod_id, api_key, args.dry_run)
        push_project(endpoint, args.dry_run)
        print("bootstrapping pod environment (venv on /workspace, deps)...")
        rc = remote_exec(endpoint, f"bash {REMOTE_DIR}/podenv/bootstrap.sh",
                         args.dry_run)
        if rc == 42:
            # Broken GPU host (cuInit fails). A stopped pod would restart on
            # the same host — terminate so the next run lands elsewhere.
            print("broken GPU host — terminating pod; rerun to get a fresh host",
                  file=sys.stderr)
            terminate_pod(pod_id, api_key, args.dry_run)
            clear_state()
            terminated = True
        elif rc != 0:
            print("error: bootstrap failed", file=sys.stderr)
        else:
            job = (f"cd {REMOTE_DIR} && source {REMOTE_VENV}/bin/activate && "
                   f"export HF_HOME={REMOTE_HF_HOME} && {JOBS[args.job]}")
            print(f"running on pod: {JOBS[args.job]}")
            exit_code = remote_exec(endpoint, job, args.dry_run)
            pull_artifacts(endpoint, args.dry_run)
    finally:
        if terminated:
            pass
        elif args.keep_up:
            print("\n--keep-up: leaving the pod running (billing continues!)")
        else:
            # ALWAYS stop the pod, even on failure/Ctrl-C.
            print("\nstopping pod...")
            try:
                stop_pod(pod_id, api_key, args.dry_run)
            except Exception as e:
                print(f"warning: failed to stop pod {pod_id}: {e} — "
                      f"check the RunPod console!", file=sys.stderr)
    sys.exit(exit_code)


def cmd_down(args):
    api_key = require_api_key(args.dry_run)
    pod_id = load_state().get("pod_id")
    if not pod_id:
        print("no saved pod id; nothing to stop")
        return
    print(f"stopping pod {pod_id}...")
    stop_pod(pod_id, api_key, args.dry_run)
    print("stopped (volume persists; `up` restarts it)")


def cmd_destroy(args):
    api_key = require_api_key(args.dry_run)
    pod_id = load_state().get("pod_id")
    if not pod_id:
        print("no saved pod id; nothing to destroy")
        return
    if not args.dry_run:
        reply = input(f"terminate pod {pod_id} ({POD_NAME}) permanently? [y/N] ")
        if reply.strip().lower() != "y":
            print("aborted")
            return
    terminate_pod(pod_id, api_key, args.dry_run)
    if not args.dry_run:
        clear_state()
    print("terminated (volume and its caches are gone)")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="print requests/commands instead of executing")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("up", help="ensure pod exists+running, wait for SSH")
    sub.add_parser("status", help="pod state, cost, SSH endpoint")
    sub.add_parser("push", help="rsync code + data to the pod")
    sub.add_parser("pull", help="fetch checkpoints/ + eval outputs")
    sub.add_parser("shell", help="interactive SSH into the pod")
    p_run = sub.add_parser("run", help="up -> push -> job -> pull -> down, always")
    p_run.add_argument("job", nargs="?", default="all", choices=list(JOBS))
    p_run.add_argument("--keep-up", action="store_true",
                       help="leave the pod running after the job")
    sub.add_parser("down", help="stop the pod (volume persists)")
    sub.add_parser("destroy", help="terminate the pod entirely (y/N confirm)")

    args = parser.parse_args()
    if not hasattr(args, "keep_up"):
        args.keep_up = False
    {"up": cmd_up, "status": cmd_status, "push": cmd_push, "pull": cmd_pull,
     "shell": cmd_shell, "run": cmd_run, "down": cmd_down,
     "destroy": cmd_destroy}[args.cmd](args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)
