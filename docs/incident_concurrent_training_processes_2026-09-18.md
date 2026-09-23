# Incident: Six Concurrent Training Processes, ~23:24-01:30 (2026-09-17/18)

## What happened

Over roughly two hours, six separate instances of `scripts/train_transfer_cnn.py
--model efficientnet_b0` ended up running **at the same time**, all writing to the
same shared output paths (`experiments/efficientnet_b0/{checkpoints,history}/`),
plus 8 orphaned `multiprocessing` worker processes left over from one abandoned
attempt. None of this was intentional -- each relaunch was meant to *replace* the
previous attempt, not run alongside it.

## Timeline of relaunches (none of which actually killed the previous one)

| Time (approx) | Action | What actually happened |
|---|---|---|
| 23:24 | First launch (`num_workers=0`, piped through `tee`) | Started (PID 6596). Appeared to produce zero output due to stdout buffering through the pipe; assumed dead. |
| 23:55 | Relaunch with `python -u`, direct redirect | Started (PID 5116). Previous process (6596) was never actually killed -- still running. |
| 00:11 | Relaunch with `num_workers=4` to speed up data loading | Started (PID 11364), spawned 8 worker subprocesses. This attempt destabilized Windows multiprocessing and was believed to have crashed; reverted the code change. Neither the parent nor its 8 workers were actually killed. |
| ~00:17-00:24 | Three more relaunches with the reverted (`num_workers=0`) code, each assumed to have failed based on `tasklist` showing no process | Three more processes started (PIDs 12504, 8040, 18172). None of the earlier ones had actually died. |

By 01:27, **six full training processes plus 8 orphaned workers (14 total)** were
running concurrently, all competing for this machine's 8 CPU cores and 8GB RAM,
and all writing to the same checkpoint/history files.

## Why this went undetected for two hours

Every "is it still running?" check in this window used `tasklist` (both as a
plain command and, later, with `//V`). **`tasklist` never showed any of these
processes, at any point** -- including when a controlled test proved a
background process was genuinely alive and running. This produced a false
impression that every relaunch was starting from a clean slate, when in fact
each one was piling onto the last.

The check that actually worked, once tried:

```powershell
Get-CimInstance Win32_Process -Filter "Name = 'python3.11.exe'" |
    Select-Object ProcessId, CreationDate, CommandLine
```

This immediately listed all 14 processes with full command lines, making it
possible to distinguish the real training runs (`train_transfer_cnn.py
--model efficientnet_b0`) from the orphaned `multiprocessing.spawn` workers
(`--multiprocessing-fork`, tagged with their parent PID).

**Lesson: on this machine/environment, `tasklist` is not a reliable way to check
whether a backgrounded Python process is still alive. `Get-CimInstance
Win32_Process` (via PowerShell, with sandboxing disabled) is.** Any future
"is training still running" check should use this method, not `tasklist`.

## How the corruption was actually noticed

Not from a crash or an error message -- from a **data inconsistency**. The
training script's own stdout log (`logs_train_efficientnet_b0.txt`, one line per
epoch) and its incrementally-saved `training_history_live.json` (written fresh
after every epoch, added specifically so progress would survive a crash) should
always describe the *same* run and therefore agree. At one check, epoch 1's
numbers in the log file (1048.4s, val_acc=0.7198) did not match epoch 1's numbers
in the JSON file (742.2s, val_acc=0.7137) -- two different values for what
should have been one fixed, already-completed epoch. That mismatch is what's
only possible if two different processes are independently writing "epoch 1" to
the same shared files. This surfaced the whole incident.

Incidentally, this validates the incremental-history-write design added earlier
in the evening (writing `training_history_live.json` after every epoch instead
of only at the end) -- it was added so a crash wouldn't lose progress, but it
also happened to be the thing that exposed this unrelated bug.

## Resolution

1. Listed all `python3.11.exe` processes with `Get-CimInstance Win32_Process`
   (14 found: 6 training runs + 8 orphaned workers).
2. Killed all 14 with `Stop-Process -Id <pid> -Force`.
3. Re-ran `Get-CimInstance` to confirm zero Python processes remained.
4. Deleted all contents of `experiments/efficientnet_b0/{checkpoints,history,
   config,plots}/` and the stdout log -- every number produced during the
   overlap window is untrustworthy and was discarded rather than salvaged.
5. Started exactly one fresh training process, then immediately re-ran
   `Get-CimInstance` again to confirm only one PID exists before trusting any
   further progress numbers from it.

**Every EfficientNet-B0 training number reported earlier in this session
(72.0%, 75.8%, "7 epochs," etc.) came from this contaminated window and should
be treated as void.** Training restarted from epoch 1 with a single verified
process.

## Process going forward

- Before relaunching any background training job, verify the previous one is
  actually dead via `Get-CimInstance Win32_Process`, not `tasklist`.
- After launching, immediately re-check the process list to confirm exactly one
  matching process exists before trusting any output from it.
- If a background launch appears to produce no output, do not assume it crashed
  and relaunch -- check the process list first. On this machine, `python -u`
  with a direct redirect can still take a long time to show its first print
  (library imports alone measured at ~37s under load), which looks identical to
  "silently dead" if you only look at the log file.
- Keep writing `training_history_live.json` incrementally (one write per
  epoch) -- it is cheap, and it is what caught this bug.
