"""
Part 2 architecture comparison (Stage A / stress), model 4 of 5: CNN-LSTM on raw BVP.

REVISION 2026-09-18 (v2): the first version of this script trained a fixed 12 epochs with
a flat learning rate and no early stopping. Result: F1=0 for the first 5 epochs (model
hadn't crossed the 0.5 threshold yet, though AUC was already climbing -- not itself a
problem), then a real instability problem -- val AUC peaked at epoch 8 (0.928) and then
dropped sharply at epoch 9 (0.785) before partially recovering. That's a training-setup
problem (no LR decay to let the model settle near a good solution), not evidence the
approach doesn't work. This revision adds, and only adds, the fixes that address that:
LR scheduling, early stopping on the metric that actually matters here (AUC, not loss),
gradient clipping, a bit more regularization, explicit subject-split verification, and a
post-hoc threshold analysis -- same data pipeline, same model family, same test-set
isolation as before.

SIGNAL PREPROCESSING: unchanged from v1 -- standard PPG bandpass (0.5-8 Hz, Butterworth,
zero-phase) applied to WESAD's raw 64 Hz BVP, since no filtering existed anywhere upstream
in this project before this comparison.

TEST-SET ISOLATION: unchanged and, if anything, stricter than v1. TRAIN_SUBJECTS and
INTERNAL_VAL_SUBJECTS are the only data ever loaded into memory by main(). TEST_SUBJECTS'
IDENTITIES are printed (to verify zero overlap) but their signal data is never read by this
script. evaluate_final_test() is defined at the bottom for later use and is NOT called from
main() or anywhere else in this file -- running it is a separate, explicit action.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score, precision_score,
                              recall_score, average_precision_score, confusion_matrix)

from paths import ROOT
from wesad_comparison_split import (TRAIN_SUBJECTS, TEST_SUBJECTS,
                                     INTERNAL_TRAIN_SUBJECTS, INTERNAL_VAL_SUBJECTS)

RAW_PATH = ROOT / "dataset" / "WESAD" / "wesad_raw_windows.npz"
OUT_DIR = ROOT / "experiments" / "wesad_stress_comparison"
PLOTS_DIR = OUT_DIR / "plots"
BVP_FS = 64
SEED = 42
MAX_EPOCHS = 30          # ceiling; early stopping (patience 5 on val AUC) is expected to
                          # cut this short well before 30, same pattern as the image models
BATCH_SIZE = 16
INITIAL_LR = 1e-3        # unchanged from v1 -- no evidence v1's LR itself was wrong, only
                          # that it never decayed
WEIGHT_DECAY = 1e-4      # L2, new in v2
GRAD_CLIP_NORM = 1.0     # new in v2
EARLY_STOP_PATIENCE = 5
LR_PATIENCE = 2
LR_FACTOR = 0.5
MIN_LR = 1e-6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)


def bandpass_bvp(x, fs=BVP_FS, low=0.5, high=8.0, order=3):
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x, axis=-1).astype(np.float32)


def verify_subject_split():
    """Print and check the subject-level split exactly as requested -- this is an explicit
    runtime check, not just the assertions already in wesad_comparison_split.py."""
    train_set = set(INTERNAL_TRAIN_SUBJECTS)
    val_set = set(INTERNAL_VAL_SUBJECTS)
    test_set = set(TEST_SUBJECTS)
    print("Training subjects:  ", sorted(train_set))
    print("Validation subjects:", sorted(val_set))
    print("Test subjects:      ", sorted(test_set), "(identities only -- their signal data "
          "is never loaded by this script)")
    overlap = (train_set & val_set) | (train_set & test_set) | (val_set & test_set)
    print("Overlap:            ", sorted(overlap) if overlap else "none")
    assert not overlap, f"Subject split is not independent -- overlap: {overlap}"
    print("Subject-level split verified independent.\n")


def load_signal(subjects):
    """Loads BVP windows for exactly the given subjects. Never called with TEST_SUBJECTS
    in this file."""
    d = np.load(RAW_PATH, allow_pickle=True)
    mask = np.isin(d["subject"], subjects)
    bvp = bandpass_bvp(d["BVP"][mask, :, 0])
    bvp = (bvp - bvp.mean(axis=1, keepdims=True)) / (bvp.std(axis=1, keepdims=True) + 1e-8)
    return bvp, d["label"][mask].astype(np.float32)


class CNNLSTM(nn.Module):
    """v2: 2-layer LSTM with inter-layer dropout=0.2 (PyTorch's nn.LSTM `dropout` argument
    only applies between stacked layers -- there is no native equivalent to Keras'
    per-timestep `recurrent_dropout` for a plain LSTM, so that specific knob is
    approximated this way rather than faked). Head dropout unchanged at 0.3. L2 applied via
    the optimizer's weight_decay, not here."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=32, num_layers=2,
                             batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.3), nn.Linear(16, 1))

    def forward(self, x):
        h = self.conv(x)
        h = h.transpose(1, 2)
        _, (hn, _) = self.lstm(h)
        return self.head(hn[-1])


def run_epoch(model, X, y, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    idx = np.random.permutation(len(X)) if is_train else np.arange(len(X))
    criterion = nn.BCEWithLogitsLoss()
    total_loss, all_probs, all_labels = 0.0, [], []
    with torch.set_grad_enabled(is_train):
        for start in range(0, len(idx), BATCH_SIZE):
            batch_idx = idx[start:start + BATCH_SIZE]
            xb = torch.from_numpy(X[batch_idx]).unsqueeze(1).to(DEVICE)
            yb = torch.from_numpy(y[batch_idx]).float().unsqueeze(1).to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
            total_loss += loss.item() * len(batch_idx)
            all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy().ravel().tolist())
            all_labels.extend(yb.cpu().numpy().ravel().tolist())
    return total_loss / len(idx), np.array(all_labels), np.array(all_probs)


def clean_eval(model, X, y):
    """No-grad, eval-mode pass -- used to get a clean (non-augmented-by-dropout) AUC
    reading on the training set itself, for the train-vs-val diagnostic plots."""
    return run_epoch(model, X, y, optimizer=None)


def threshold_sweep(y_true, y_prob):
    """Validation-only threshold analysis, per spec. Never called with test data."""
    rows = []
    for t in np.arange(0.10, 0.91, 0.05):
        pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        rows.append(dict(threshold=round(float(t), 2), f1=f1, precision=precision,
                          recall=recall, specificity=specificity))
    best = max(rows, key=lambda r: r["f1"])
    return rows, best


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    verify_subject_split()

    X_train, y_train = load_signal(INTERNAL_TRAIN_SUBJECTS)
    X_val, y_val = load_signal(INTERNAL_VAL_SUBJECTS)
    print(f"Train windows: {len(X_train)}  Val windows: {len(X_val)}\n")

    model = CNNLSTM().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LR)

    history = []
    best_val_auc, best_state, best_epoch = -1, None, None
    epochs_without_improvement = 0

    print(f"{'Epoch':<7}{'TrainLoss':<12}{'ValLoss':<10}{'TrainAUC':<10}{'ValAUC':<10}{'LR'}")
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, _, _ = run_epoch(model, X_train, y_train, optimizer)
        _, y_train_true, y_train_prob = clean_eval(model, X_train, y_train)
        val_loss, y_val_true, y_val_prob = run_epoch(model, X_val, y_val)

        train_auc = roc_auc_score(y_train_true, y_train_prob)
        val_auc = roc_auc_score(y_val_true, y_val_prob)
        val_pred = (y_val_prob >= 0.5).astype(int)
        val_f1 = f1_score(y_val_true, val_pred, zero_division=0)
        val_acc = accuracy_score(y_val_true, val_pred)
        val_precision = precision_score(y_val_true, val_pred, zero_division=0)
        val_recall = recall_score(y_val_true, val_pred, zero_division=0)
        val_pr_auc = average_precision_score(y_val_true, y_val_prob)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"{epoch:<7}{train_loss:<12.4f}{val_loss:<10.4f}{train_auc:<10.4f}"
              f"{val_auc:<10.4f}{current_lr:.2e}")

        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                             train_auc=train_auc, val_auc=val_auc, val_f1=val_f1,
                             val_acc=val_acc, val_precision=val_precision,
                             val_recall=val_recall, val_pr_auc=val_pr_auc, lr=current_lr))

        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            print(f"  -> new best (val_auc={best_val_auc:.4f}), checkpoint saved")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"  -> early stopping at epoch {epoch} "
                      f"(no val_auc improvement for {EARLY_STOP_PATIENCE} epochs)")
                break

    model.load_state_dict(best_state)  # restore best-val-AUC weights, per spec
    torch.save(model.state_dict(), OUT_DIR / "cnn_lstm_v2.pt")
    with open(OUT_DIR / "cnn_lstm_v2_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # --- diagnostic plots ---
    epochs_x = [h["epoch"] for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(epochs_x, [h["train_loss"] for h in history], label="Train")
    axes[0, 0].plot(epochs_x, [h["val_loss"] for h in history], label="Val")
    axes[0, 0].axvline(best_epoch, color="gray", linestyle="--", label=f"Best (ep {best_epoch})")
    axes[0, 0].set_title("Loss"); axes[0, 0].set_xlabel("Epoch"); axes[0, 0].legend()

    axes[0, 1].plot(epochs_x, [h["train_auc"] for h in history], label="Train")
    axes[0, 1].plot(epochs_x, [h["val_auc"] for h in history], label="Val")
    axes[0, 1].axvline(best_epoch, color="gray", linestyle="--")
    axes[0, 1].set_title("ROC-AUC"); axes[0, 1].set_xlabel("Epoch"); axes[0, 1].legend()

    axes[1, 0].plot(epochs_x, [h["val_f1"] for h in history], label="Val F1 (0.5 threshold)")
    axes[1, 0].axvline(best_epoch, color="gray", linestyle="--")
    axes[1, 0].set_title("Validation F1 (fixed 0.5 threshold)")
    axes[1, 0].set_xlabel("Epoch"); axes[1, 0].legend()

    axes[1, 1].plot(epochs_x, [h["lr"] for h in history])
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Learning rate"); axes[1, 1].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cnn_lstm_v2_training_diagnostics.png", dpi=150)
    plt.close(fig)

    # --- threshold analysis, validation only ---
    _, y_val_true, y_val_prob = clean_eval(model, X_val, y_val)
    sweep, best_thresh = threshold_sweep(y_val_true, y_val_prob)
    print("\n=== Validation threshold sweep (best-val-AUC checkpoint) ===")
    print(f"{'Thresh':<8}{'F1':<8}{'Precision':<11}{'Recall':<9}{'Specificity'}")
    for r in sweep:
        marker = "  <-- best F1" if r["threshold"] == best_thresh["threshold"] else ""
        print(f"{r['threshold']:<8}{r['f1']:<8.3f}{r['precision']:<11.3f}"
              f"{r['recall']:<9.3f}{r['specificity']:.3f}{marker}")

    with open(OUT_DIR / "cnn_lstm_v2_threshold_sweep.json", "w") as f:
        json.dump(dict(sweep=sweep, best_by_f1=best_thresh), f, indent=2)

    print(f"\n=== SUMMARY (best epoch {best_epoch}, restored) ===")
    print(f"Best val AUC:  {best_val_auc:.4f}")
    print(f"Val PR-AUC (imbalance-aware, at best epoch): "
          f"{[h for h in history if h['epoch'] == best_epoch][0]['val_pr_auc']:.4f}")
    print(f"Best-F1 threshold (validation-selected): {best_thresh['threshold']} "
          f"-> F1={best_thresh['f1']:.3f} precision={best_thresh['precision']:.3f} "
          f"recall={best_thresh['recall']:.3f} specificity={best_thresh['specificity']:.3f}")
    print(f"Early stopped: {'yes' if epoch < MAX_EPOCHS else 'no (hit MAX_EPOCHS)'} "
          f"at epoch {epoch}")
    print("\nTEST_SUBJECTS were never loaded. evaluate_final_test() is defined below but "
          "NOT called -- run it explicitly, only when authorized.")


def evaluate_final_test(model_path=OUT_DIR / "cnn_lstm_v2.pt", threshold=0.5):
    """NOT called automatically. This is the only place in this file allowed to load
    TEST_SUBJECTS' signal data. Call manually, explicitly, only once the full 5-model
    stress comparison is ready for its one official test-set read (same gate discipline as
    scripts/eval_transfer_cnn_finalists.py on the image side)."""
    X_test, y_test = load_signal(TEST_SUBJECTS)
    model = CNNLSTM().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    _, y_true, y_prob = clean_eval(model, X_test, y_test)
    pred = (y_prob >= threshold).astype(int)
    return dict(
        n=len(y_true), threshold=threshold,
        accuracy=accuracy_score(y_true, pred),
        roc_auc=roc_auc_score(y_true, y_prob),
        pr_auc=average_precision_score(y_true, y_prob),
        f1=f1_score(y_true, pred, zero_division=0),
        precision=precision_score(y_true, pred, zero_division=0),
        recall=recall_score(y_true, pred, zero_division=0),
    )


if __name__ == "__main__":
    main()
