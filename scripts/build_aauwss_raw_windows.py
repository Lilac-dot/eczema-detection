"""
Build windowed signal tensors from the AAUWSS overnight-sleep dataset's Empatica E4
wearable exports, aligned to the dataset's own manually-scored AASM sleep-stage
annotations (30-second epochs -- the standard epoch length used throughout sleep
medicine, and the granularity AAUWSS's own annotators used).

Dataset layout expected (see DATASETS.md): dataset/AAUWSS/empatica/subject_<NN>/{EDA,TEMP,
BVP,ACC,HR,IBI,tags}.csv (standard Empatica E4 export format: row 1 = session start as a
Unix timestamp, row 2 = sample rate in Hz, then samples) and
dataset/AAUWSS/annotations/subject_<NN>_manual_annotation.xlsx (columns: index, "Event
Start Time" [naive datetime], "Sleep Stage" [Wake/N1/N2/N3/REM], "Epoch Number",
"Subject").

Timestamp alignment: the annotation sheet's "Event Start Time" was checked against each
subject's own E4 recording window (Unix timestamps interpreted as UTC) and found to line
up correctly with no timezone offset needed -- e.g. subject 1's first annotated epoch
falls ~2.3 hours after the E4 file's own start time, a plausible device-on-before-sleep
gap, not a negative or implausibly large one. Several later subjects (dated after the
2023-03-26 EU daylight-saving switch) have annotated epochs that run past their E4
file's recorded end; checking all four E4 modalities for one such subject (S12) confirms
every modality starts and stops within the same second of each other, meaning the wrist
device itself stopped recording early (most likely a flat battery, since these are
unattended overnight recordings) rather than a timestamp misalignment. This is handled
below by simply dropping any epoch whose 30-second window is not fully covered by every
modality's recorded range, not by adjusting timestamps.

Label design, deliberately mirroring build_wesad_raw_windows.py's condition/label split:
  label = 1 if the epoch's AASM stage is "Wake" (the sleep-disruption/positive class,
          analogous to WESAD's stress=1), else 0 (asleep: N1, N2, N3, or REM).
  condition = 1 if asleep (N1/N2/N3/REM) -- this is the CALM/BASELINE reference state a
          subject is calibrated against, the sleep-domain analogue of WESAD's calm
          pre-stress baseline period -- else 0 (Wake). Using condition==1 as the baseline
          marker matches wesad_calibration.compute_subject_baseline_stats()'s
          baseline_condition=1 default exactly, so that helper is reusable unmodified.

Writes dataset/AAUWSS/aauwss_raw_windows.npz with arrays:
  EDA (N,120)  TEMP (N,120)  BVP (N,1920)  ACC (N,960,3)  label (N,)  condition (N,)
  subject (N,) [strings, "subject_01".."subject_13"]  stage (N,) [strings, raw AASM label]
"""
import csv
import datetime

import numpy as np
import openpyxl

from paths import AAUWSS_DIR

OUT_NPZ = AAUWSS_DIR / "aauwss_raw_windows.npz"

WINDOW_SEC = 30
FS = {"EDA": 4, "TEMP": 4, "BVP": 64, "ACC": 32}
WAKE_STAGE = "Wake"
SUBJECT_IDS = [f"{i:02d}" for i in range(1, 14)]


def load_e4_csv(path, is_acc=False):
    with open(path, newline="") as f:
        r = csv.reader(f)
        start_ts = float(next(r)[0].strip())
        rate = float(next(r)[0].strip())
        rows = [row for row in r if row]
    if is_acc:
        data = np.array([[float(v) for v in row[:3]] for row in rows], dtype=np.float32)
        data /= 64.0  # E4's native ACC unit is 1/64 g -- convert to g (info.txt)
    else:
        data = np.array([float(row[0]) for row in rows], dtype=np.float32)
    return start_ts, rate, data


def load_annotations(path):
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb["Sheet1"]
    epochs = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        _, event_start, stage, epoch_num, _subj = row[:5]
        if event_start is None or stage is None:
            continue
        epochs.append((event_start, stage))
    return epochs


def process_subject(sid):
    ann_path = AAUWSS_DIR / "annotations" / f"subject_{sid}_manual_annotation.xlsx"
    epochs = load_annotations(ann_path)

    e4_dir = AAUWSS_DIR / "empatica" / f"subject_{sid}"
    signals, starts, rates = {}, {}, {}
    for mod, fs in FS.items():
        start_ts, rate, data = load_e4_csv(e4_dir / f"{mod}.csv", is_acc=(mod == "ACC"))
        signals[mod], starts[mod], rates[mod] = data, start_ts, rate

    rows = {"EDA": [], "TEMP": [], "BVP": [], "ACC": [], "label": [], "condition": [],
            "subject": [], "stage": []}
    n_skipped = 0
    for event_start, stage in epochs:
        epoch_start_ts = event_start.replace(tzinfo=datetime.timezone.utc).timestamp()
        epoch_end_ts = epoch_start_ts + WINDOW_SEC

        ok = True
        window_data = {}
        for mod, fs in FS.items():
            n_needed = WINDOW_SEC * fs
            s = int(round((epoch_start_ts - starts[mod]) * fs))
            e = s + n_needed
            if s < 0 or e > len(signals[mod]):
                ok = False
                break
            window_data[mod] = signals[mod][s:e]
        if not ok:
            n_skipped += 1
            continue

        for mod in FS:
            rows[mod].append(window_data[mod].astype(np.float32))
        rows["label"].append(int(stage == WAKE_STAGE))
        rows["condition"].append(int(stage != WAKE_STAGE))  # 1 = asleep = baseline/calm
        rows["subject"].append(f"subject_{sid}")
        rows["stage"].append(stage)

    return rows, n_skipped


def main():
    all_rows = {"EDA": [], "TEMP": [], "BVP": [], "ACC": [], "label": [], "condition": [],
                "subject": [], "stage": []}
    for sid in SUBJECT_IDS:
        rows, n_skipped = process_subject(sid)
        n = len(rows["label"])
        n_wake = sum(rows["label"])
        print(f"  subject_{sid}: {n} windows kept, {n_skipped} dropped "
              f"(outside E4 recording range), {n_wake} Wake "
              f"({100 * n_wake / n if n else 0:.1f}%)")
        for k in all_rows:
            all_rows[k].extend(rows[k])

    EDA = np.stack(all_rows["EDA"])
    TEMP = np.stack(all_rows["TEMP"])
    BVP = np.stack(all_rows["BVP"])
    ACC = np.stack(all_rows["ACC"])
    label = np.array(all_rows["label"], dtype=np.int64)
    condition = np.array(all_rows["condition"], dtype=np.int64)
    subject = np.array(all_rows["subject"])
    stage = np.array(all_rows["stage"])

    print(f"\nShapes: EDA={EDA.shape} TEMP={TEMP.shape} BVP={BVP.shape} ACC={ACC.shape}")
    print(f"Total windows: {len(label)}  Wake: {label.sum()} ({100 * label.mean():.1f}%)")
    print(f"Asleep/baseline-condition windows (for per-subject calibration): "
          f"{(condition == 1).sum()}")

    np.savez_compressed(OUT_NPZ, EDA=EDA, TEMP=TEMP, BVP=BVP, ACC=ACC, label=label,
                         condition=condition, subject=subject, stage=stage)
    print(f"Saved: {OUT_NPZ}")


if __name__ == "__main__":
    main()
