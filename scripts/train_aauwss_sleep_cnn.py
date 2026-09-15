"""
Train the "complex" AAUWSS sleep-disruption model: a 4-branch 1D-CNN (one branch per
modality -- EDA, TEMP, BVP, ACC) with a learned attention-gated fusion layer, instead of
LightGBM on hand-crafted stats (train_aauwss_sleep.py). Mirrors
train_wesad_cnn_attention.py exactly -- same architecture, same per-subject baseline
normalization approach, same LOSO-CV protocol, same seed-ensembling -- since this is
being tried specifically to check whether train_aauwss_sleep.py's near-chance LightGBM
result (mean AUC ~0.46) reflects a genuine ceiling on what 30-second-epoch, hand-crafted
window statistics can extract from this dataset, or whether a model that reads the raw
signal directly does better. See docs/aauwss_sleep_model_results_2026-09-14.md for the
full comparison once both are run.

13 subjects (vs. WESAD's 15). condition==1 marks each window's baseline/calibration
reference state -- here "asleep" (N1/N2/N3/REM), not "calm pre-task" -- so
wesad_calibration.compute_subject_baseline_stats()'s baseline_condition=1 default applies
unmodified (see build_aauwss_raw_windows.py's docstring for the label/condition design).

Writes:
  dataset/AAUWSS/aauwss_cnn_loso_fold_results.csv
  models/aauwss_sleep_cnn_attention.pt  (final model, trained on all 13 subjects)
"""
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from paths import ROOT, AAUWSS_DIR
from loso_report import print_reliability_summary
from wesad_calibration import compute_subject_baseline_stats, normalize_by_subject

NPZ_PATH = AAUWSS_DIR / "aauwss_raw_windows.npz"
FOLD_RESULTS_CSV = AAUWSS_DIR / "aauwss_cnn_loso_fold_results.csv"
SEED = 42
EMBED_DIM = 32
MAX_EPOCHS = 30
PATIENCE = 6
# Increased from WESAD's BATCH_SIZE=32 to 256: AAUWSS has ~4.5x more windows per fold
# (~8200 vs ~1800 training windows), so at batch=32 each fold trains at ~256
# Python-loop-bound optimizer steps per epoch on CPU-only hardware, which measured at
# over 24 minutes without finishing even the first of 13 folds -- impractical. A larger
# batch cuts steps-per-epoch by 8x with the same total data seen per epoch; this is a
# disclosed practicality adaptation, not a tuning choice aimed at a better score.
BATCH_SIZE = 256
LR = 1e-3
# Reduced from WESAD's N_ENSEMBLE=3 to 1: this machine has 8GB total RAM with well under
# 1GB free even before training starts (many background apps), and AAUWSS has ~4.5x more
# windows per fold than WESAD's CNN was trained on (9700 vs ~2140) -- a 3-model ensemble
# on this much data triggered an out-of-memory kill. This is a disclosed hardware-driven
# adaptation, not a modeling choice; see docs/aauwss_sleep_model_results_2026-09-14.md.
N_ENSEMBLE = 1

torch.manual_seed(SEED)
np.random.seed(SEED)


class ModalityBranch(nn.Module):
    def __init__(self, in_channels, embed_dim=EMBED_DIM, base=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, base, kernel_size=7, padding=3), nn.BatchNorm1d(base), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(base, base * 2, kernel_size=5, padding=2), nn.BatchNorm1d(base * 2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(base * 2, embed_dim, kernel_size=3, padding=1), nn.BatchNorm1d(embed_dim), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class AttentionFusion(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, n_modalities=4):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * n_modalities, n_modalities * 4), nn.ReLU(),
            nn.Linear(n_modalities * 4, n_modalities),
        )

    def forward(self, embeds):
        b, m, d = embeds.shape
        flat = embeds.reshape(b, m * d)
        weights = torch.softmax(self.gate(flat), dim=-1)
        fused = (embeds * weights.unsqueeze(-1)).sum(dim=1)
        return fused, weights


class AauwssSleepNet(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.eda_branch = ModalityBranch(1, embed_dim)
        self.temp_branch = ModalityBranch(1, embed_dim)
        self.bvp_branch = ModalityBranch(1, embed_dim)
        self.acc_branch = ModalityBranch(3, embed_dim)
        self.fusion = AttentionFusion(embed_dim, n_modalities=4)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 16), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(16, 1),
        )

    def forward(self, eda, temp, bvp, acc):
        e = self.eda_branch(eda)
        t = self.temp_branch(temp)
        b = self.bvp_branch(bvp)
        a = self.acc_branch(acc)
        embeds = torch.stack([e, t, b, a], dim=1)
        fused, weights = self.fusion(embeds)
        logit = self.classifier(fused).squeeze(-1)
        return logit, weights


def load_data():
    npz = np.load(NPZ_PATH, allow_pickle=True)
    EDA = npz["EDA"]      # (N, 120) -- no trailing singleton dim in this npz, unlike WESAD's
    TEMP = npz["TEMP"]    # (N, 120)
    BVP = npz["BVP"]      # (N, 1920)
    ACC = npz["ACC"]      # (N, 960, 3)
    label = npz["label"]
    condition = npz["condition"]
    subject = npz["subject"]
    return EDA, TEMP, BVP, ACC, label, condition, subject


def to_tensors(idx, EDA, TEMP, BVP, ACC, label, subject, stats):
    subj_ids = subject[idx]
    e = normalize_by_subject(EDA[idx], subj_ids, stats, "EDA")
    t = normalize_by_subject(TEMP[idx], subj_ids, stats, "TEMP")
    b = normalize_by_subject(BVP[idx], subj_ids, stats, "BVP")
    a = normalize_by_subject(ACC[idx], subj_ids, stats, "ACC")
    e = torch.tensor(e, dtype=torch.float32).unsqueeze(1)
    t = torch.tensor(t, dtype=torch.float32).unsqueeze(1)
    b = torch.tensor(b, dtype=torch.float32).unsqueeze(1)
    a = torch.tensor(a, dtype=torch.float32).permute(0, 2, 1)
    y = torch.tensor(label[idx], dtype=torch.float32)
    return e, t, b, a, y


def best_f1_threshold(probs, y):
    candidates = np.linspace(0.01, 0.99, 197)
    best_t, best_f1 = 0.5, -1.0
    for th in candidates:
        pred = (probs >= th).astype(int)
        tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum(); fn = ((pred == 0) & (y == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, th
    return best_t, best_f1


def evaluate(probs, y, threshold):
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (y == 1)).sum()); tn = int(((preds == 0) & (y == 0)).sum())
    fp = int(((preds == 1) & (y == 0)).sum()); fn = int(((preds == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (preds == y).mean()
    return dict(acc=acc, tp=tp, tn=tn, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)


def train_one_fold(train_tensors, val_tensors, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    e_tr, t_tr, b_tr, a_tr, y_tr = train_tensors
    e_va, t_va, b_va, a_va, y_va = val_tensors

    model = AauwssSleepNet()
    n_pos, n_neg = y_tr.sum().item(), (1 - y_tr).sum().item()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    n = len(y_tr)
    best_val_auc, best_state, epochs_no_improve = -1.0, None, 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, BATCH_SIZE):
            batch_idx = perm[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            logits, _ = model(e_tr[batch_idx], t_tr[batch_idx], b_tr[batch_idx], a_tr[batch_idx])
            loss = criterion(logits, y_tr[batch_idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits, _ = model(e_va, t_va, b_va, a_va)
            val_probs = torch.sigmoid(val_logits).numpy()
        val_auc = roc_auc_score(y_va.numpy(), val_probs) if len(np.unique(y_va.numpy())) > 1 else 0.5

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            break

    model.load_state_dict(best_state)
    return model, best_val_auc


def train_ensemble(train_tensors, val_tensors, n_models=N_ENSEMBLE, seed_base=0):
    """train_tensors/val_tensors are built ONCE by the caller and reused across all
    n_models members -- rebuilding them per member (the original approach) meant holding
    multiple redundant copies of the full per-fold tensor set in memory at once, which is
    what triggered the OOM kill on this machine at AAUWSS's window count. See N_ENSEMBLE's
    comment above."""
    models, val_aucs = [], []
    for k in range(n_models):
        model, val_auc = train_one_fold(train_tensors, val_tensors,
                                         seed=SEED + seed_base * 1000 + k)
        models.append(model)
        val_aucs.append(val_auc)
    return models, val_aucs


def ensemble_predict(models, e, t, b, a):
    probs_list, weights_list = [], []
    with torch.no_grad():
        for model in models:
            model.eval()
            logits, weights = model(e, t, b, a)
            probs_list.append(torch.sigmoid(logits).numpy())
            weights_list.append(weights.numpy())
    return np.mean(probs_list, axis=0), np.mean(weights_list, axis=0)


def main():
    EDA, TEMP, BVP, ACC, label, condition, subject = load_data()
    subjects = sorted(np.unique(subject))
    print(f"Subjects: {len(subjects)} -> {subjects}")
    print(f"Windows: {len(label)}  Wake: {label.sum()} ({100*label.mean():.1f}%)")

    stats = compute_subject_baseline_stats(EDA, TEMP, BVP, ACC, condition, subject)

    fold_rows = []
    pooled_probs, pooled_y = [], []
    all_weights = []

    for i, test_subj in enumerate(subjects):
        remaining = [s for s in subjects if s != test_subj]
        val_subj = remaining[i % len(remaining)]
        train_subjs = [s for s in remaining if s != val_subj]

        train_idx = np.isin(subject, train_subjs)
        val_idx = subject == val_subj
        test_idx = subject == test_subj

        train_tensors = to_tensors(train_idx, EDA, TEMP, BVP, ACC, label, subject, stats)
        val_tensors = to_tensors(val_idx, EDA, TEMP, BVP, ACC, label, subject, stats)
        test_tensors = to_tensors(test_idx, EDA, TEMP, BVP, ACC, label, subject, stats)

        models, val_aucs = train_ensemble(train_tensors, val_tensors, seed_base=i)

        e_va, t_va, b_va, a_va, y_va = val_tensors
        e_te, t_te, b_te, a_te, y_te = test_tensors

        val_probs, _ = ensemble_predict(models, e_va, t_va, b_va, a_va)
        test_probs, test_weights = ensemble_predict(models, e_te, t_te, b_te, a_te)

        thr, val_f1 = best_f1_threshold(val_probs, y_va.numpy())
        y_te_np = y_te.numpy()
        test_auc = roc_auc_score(y_te_np, test_probs) if len(np.unique(y_te_np)) > 1 else float("nan")
        result = evaluate(test_probs, y_te_np, thr)
        result.update(subject=test_subj, val_subject=val_subj, threshold=thr,
                       val_f1=val_f1, val_auc=float(np.mean(val_aucs)), test_auc=test_auc,
                       n_test=len(y_te_np))
        fold_rows.append(result)
        pooled_probs.append(test_probs)
        pooled_y.append(y_te_np)
        all_weights.append(test_weights)

        print(f"[{test_subj}] val_subj={val_subj} thr={thr:.3f} "
              f"test_auc={test_auc:.4f} test_f1={result['f1']:.4f} test_acc={result['acc']:.4f}")

        # Explicit cleanup between folds -- this machine has very little free RAM (8GB
        # total, well under 1GB free at baseline), and 13 folds' worth of tensors/models
        # left for the garbage collector to find "eventually" was a real contributor to
        # the earlier OOM kill.
        del train_tensors, val_tensors, test_tensors, models
        gc.collect()

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(FOLD_RESULTS_CSV, index=False)

    print("\n=== CNN+Attention LOSO-CV summary across 13 folds ===")
    print(f"AUC:       mean={fold_df['test_auc'].mean():.4f}  std={fold_df['test_auc'].std():.4f}")
    print(f"F1:        mean={fold_df['f1'].mean():.4f}  std={fold_df['f1'].std():.4f}")
    print(f"Accuracy:  mean={fold_df['acc'].mean():.4f}  std={fold_df['acc'].std():.4f}")
    print(f"Precision: mean={fold_df['precision'].mean():.4f}")
    print(f"Recall:    mean={fold_df['recall'].mean():.4f}")

    print_reliability_summary(fold_df)

    pooled_probs = np.concatenate(pooled_probs)
    pooled_y = np.concatenate(pooled_y)
    pooled_thr = float(fold_df["threshold"].median())
    pooled = evaluate(pooled_probs, pooled_y, pooled_thr)
    pooled_auc = roc_auc_score(pooled_y, pooled_probs)
    print(f"\n=== Pooled (median threshold={pooled_thr:.3f}) ===")
    print(f"AUC={pooled_auc:.4f} Accuracy={pooled['acc']:.4f} "
          f"Precision={pooled['precision']:.4f} Recall={pooled['recall']:.4f} F1={pooled['f1']:.4f}")
    print(f"TN={pooled['tn']} FP={pooled['fp']} FN={pooled['fn']} TP={pooled['tp']}")

    mean_weights = np.concatenate(all_weights, axis=0).mean(axis=0)
    print(f"\nMean attention weight by modality (across all test folds):")
    for name, w in zip(["EDA", "TEMP", "BVP", "ACC"], mean_weights):
        print(f"  {name}: {w:.3f}")

    final_val_subj = subjects[-1]
    final_train_subjs = [s for s in subjects if s != final_val_subj]
    train_idx = np.isin(subject, final_train_subjs)
    val_idx = subject == final_val_subj
    final_train_tensors = to_tensors(train_idx, EDA, TEMP, BVP, ACC, label, subject, stats)
    final_val_tensors = to_tensors(val_idx, EDA, TEMP, BVP, ACC, label, subject, stats)
    final_models, _ = train_ensemble(final_train_tensors, final_val_tensors, seed_base=999)

    torch.save({
        "state_dicts": [m.state_dict() for m in final_models],
        "threshold": pooled_thr,
    }, ROOT / "models" / "aauwss_sleep_cnn_attention.pt")
    print(f"\nFinal model saved: models/aauwss_sleep_cnn_attention.pt ({N_ENSEMBLE}-model ensemble)")


if __name__ == "__main__":
    main()
