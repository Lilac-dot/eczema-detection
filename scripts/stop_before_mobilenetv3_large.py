"""One-shot watcher: waits for efficientnet_lite0 to finish (its checkpoint appearing),
then kills the overnight_queue.py process and any mobilenetv3_large training subprocess
before mobilenetv3_large can start. Run once, exits when done.

Why this exists: the user asked to skip mobilenetv3_large, but overnight_queue.py has a
fixed queue and would launch it automatically right after efficientnet_lite0 finishes.
Killing the queue's parent process is the reliable way to stop that without needing to
catch the exact moment by hand.
"""
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MARKER = PROJECT_ROOT / "experiments" / "efficientnet_lite0" / "checkpoints" / "best_overall.pt"
POLL_SECONDS = 30

print(f"Waiting for {MARKER} ...", flush=True)
while not MARKER.exists():
    time.sleep(POLL_SECONDS)
print("efficientnet_lite0 finished. Killing overnight_queue.py before mobilenetv3_large starts.",
      flush=True)

# Kill overnight_queue.py and any train_transfer_cnn.py process (in case mobilenetv3_large
# already slipped through in the narrow window between poll and this check).
ps = subprocess.run(
    ["powershell", "-NoProfile", "-Command",
     "Get-CimInstance Win32_Process -Filter \"Name LIKE 'python%'\" | "
     "Where-Object { $_.CommandLine -match 'overnight_queue.py' -or "
     "$_.CommandLine -match 'mobilenetv3_large' } | "
     "ForEach-Object { Stop-Process -Id $_.ProcessId -Force; "
     "Write-Output \"killed PID $($_.ProcessId): $($_.CommandLine)\" }"],
    capture_output=True, text=True,
)
print(ps.stdout)
print(ps.stderr)
print("Done. mobilenetv3_large will not be trained. 8-of-9 candidates are the final set "
      "for this comparison -- see docs, gate script updated accordingly.")
