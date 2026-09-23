"""Waits for the currently-running EfficientNet-B0 training process to fully finish
(both stages), then runs the rest of the model queue automatically. Built so the whole
overnight run doesn't depend on anyone being awake to launch the next step at the right
moment.

Completion signal: experiments/efficientnet_b0/checkpoints/best_overall.pt only gets
written by train_transfer_cnn.py after BOTH stages finish successfully (see that
script's main()) -- waiting for this file to exist is more robust than checking whether
a process is still running (see docs/incident_concurrent_training_processes_2026-09-18.md
for why process-liveness checks were unreliable here).

Usage:
    python overnight_queue.py
"""
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EFFICIENTNET_DONE_MARKER = PROJECT_ROOT / "experiments" / "efficientnet_b0" / "checkpoints" / "best_overall.pt"
POLL_SECONDS = 60

QUEUE = [
    "mobilenetv3_small",
    "shufflenet_v2_x0_5",
    "squeezenet1_1",
    "repghostnet_050",
    "shufflenet_v2_x1_0",
    "mobilenetv2_100",
    "efficientnet_lite0",
    "mobilenetv3_large",
]


def wait_for_efficientnet_b0():
    print(f"Waiting for {EFFICIENTNET_DONE_MARKER} to appear "
          f"(polling every {POLL_SECONDS}s)...", flush=True)
    waited = 0
    while not EFFICIENTNET_DONE_MARKER.exists():
        time.sleep(POLL_SECONDS)
        waited += POLL_SECONDS
        if waited % 600 == 0:
            print(f"  ... still waiting ({waited // 60} min elapsed)", flush=True)
    print(f"EfficientNet-B0 finished (checkpoint found after {waited // 60} min wait).",
          flush=True)


def already_done(model_name):
    marker = PROJECT_ROOT / "experiments" / model_name / "checkpoints" / "best_overall.pt"
    return marker.exists()


def main():
    wait_for_efficientnet_b0()

    print(f"\nStarting queue: {QUEUE}", flush=True)
    for i, model_name in enumerate(QUEUE, 1):
        if already_done(model_name):
            print(f"\n=== [{i}/{len(QUEUE)}] Skipping {model_name} -- "
                  f"best_overall.pt already exists ===", flush=True)
            continue
        log_path = PROJECT_ROOT / f"logs_train_{model_name}.txt"
        print(f"\n=== [{i}/{len(QUEUE)}] Starting {model_name} "
              f"({time.strftime('%H:%M:%S')}) ===", flush=True)
        with open(log_path, "w") as logf:
            result = subprocess.run(
                [sys.executable, "-u", "train_transfer_cnn.py", "--model", model_name],
                cwd=SCRIPT_DIR, stdout=logf, stderr=subprocess.STDOUT,
            )
        status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
        print(f"=== [{i}/{len(QUEUE)}] {model_name}: {status} "
              f"({time.strftime('%H:%M:%S')}) -- log: {log_path} ===", flush=True)

    print("\nOvernight queue complete. All 9 models done (or logged failures above).",
          flush=True)


if __name__ == "__main__":
    main()
