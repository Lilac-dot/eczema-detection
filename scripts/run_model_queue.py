"""Runs a fixed sequence of train_transfer_cnn.py models one at a time, each as a
BLOCKING subprocess -- this script will not start model N+1 until model N's process
has actually exited, so there is no possibility of the concurrent-process corruption
documented in docs/incident_concurrent_training_processes_2026-09-18.md.

Does NOT check for other stray train_transfer_cnn.py processes outside this queue
(e.g. one launched manually, separately) -- verify none are running (Get-CimInstance
Win32_Process, not tasklist -- see that incident doc) before starting this script.

Usage:
    python run_model_queue.py
"""
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# efficientnet_b0 deliberately excluded -- run separately, started before this queue
# existed. Order otherwise: cheapest/smallest models first, so partial results are
# available soonest if this has to be interrupted.
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


def main():
    print(f"Queue: {QUEUE}", flush=True)
    for i, model_name in enumerate(QUEUE, 1):
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

    print("\nQueue complete.", flush=True)


if __name__ == "__main__":
    main()
