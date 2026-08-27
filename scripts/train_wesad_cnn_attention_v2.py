"""
Fine-tuned version of train_wesad_cnn_attention.py, targeting the specific problem that
run showed: huge fold-to-fold variance (F1 std=0.31 across the 15 LOSO folds) despite a
mean score barely ahead of the LightGBM baseline. That variance pattern -- some folds
near-perfect, others near chance -- is the signature of OVERFITTING to the 13 training
subjects each fold, not of insufficient model capacity. So this tuning pass adds
regularization and data augmentation rather than making the network bigger/fancier;
growing the network further would very likely make the overfitting worse, not better,
at only 13 training subjects per fold.

Changes from v1, each aimed at the variance problem specifically:
  1. Data augmentation on TRAINING windows only (small Gaussian jitter + a small random
     circular time-shift per modality, applied fresh every batch) -- makes the model see a
     slightly different version of each subject's data every epoch instead of memorizing
     it outright.
  2. More dropout: 0.3 -> 0.5 in the classifier head, plus new dropout added inside each
     modality branch (was none before).
  3. Stronger weight decay: 1e-4 -> 5e-4.
  4. A learning-rate scheduler (ReduceLROnPlateau on validation AUC) instead of one fixed
     rate for the whole run, so training can take smaller steps once it's close to a good
     solution instead of overshooting.
  5. Patience raised 10 -> 15 epochs, to give the scheduler room to lower the rate and let
     training keep improving afterward, instead of stopping right before that would help.

Same LOSO-CV protocol, same data (dataset/WESAD/wesad_raw_windows.npz), same evaluation
code as v1 -- only the training procedure changed, so the comparison is fair.

Writes:
  dataset/WESAD/wesad_cnn_v2_loso_fold_results.csv
  models/wesad_stress_cnn_attention_v2.pt
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

ROOT = Path(r"C:\Users\tishy\Documents\Honors")
NPZ_PATH = ROOT / "dataset" / "WESAD" / "wesad_raw_windows.npz"
FOLD_RESULTS_CSV = ROOT / "dataset" / "WESAD" / "wesad_cnn_v2_loso_fold_results.csv"
SEED = 42
EMBED_DIM = 32
MAX_EPOCHS = 80
PATIENCE = 15
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 5e-4
NOISE_STD = 0.08          # in z-scored units (signal std=1), so this is an 8%-of-std jitter
MAX_SHIFT_FRAC = 0.10     # up to 10% of a window's length, per modality, per example

torch.manual_seed(SEED)
np.random.seed(SEED)


class ModalityBranch(nn.Module):
    def __init__(self, in_channels, embed_dim=EMBED_DIM, base=8, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, base, kernel_size=7, padding=3), nn.BatchNorm1d(base), nn.ReLU(),
            nn.Dropout(dropout), nn.MaxPool1d(2),
            nn.Conv1d(base, base * 2, kernel_size=5, padding=2), nn.BatchNorm1d(base * 2), nn.ReLU(),
            nn.Dropout(dropout), nn.MaxPool1d(2),
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


class WesadStressNet(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.eda_branch = ModalityBranch(1, embed_dim)
        self.temp_branch = ModalityBranch(1, embed_dim)
        self.bvp_branch = ModalityBranch(1, embed_dim)
        self.acc_branch = ModalityBranch(3, embed_dim)
        self.fusion = AttentionFusion(embed_dim, n_modalities=4)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 16), nn.ReLU(), nn.Dropout(0.5),
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
    EDA = npz["EDA"].squeeze(-1)
    TEMP = npz["TEMP"].squeeze(-1)
    BVP = npz["BVP"].squeeze(-1)
    ACC = npz["ACC"]
    label = npz["label"]
    subject = npz["subject"]
    return EDA, TEMP, BVP, ACC, label, subject


def to_tensors(idx, EDA, TEMP, BVP, ACC, label, eda_mu, eda_sd, temp_mu, temp_sd, bvp_mu, bvp_sd, acc_mu, acc_sd):
    e = torch.tensor((EDA[idx] - eda_mu) / eda_sd, dtype=torch.float32).unsqueeze(1)
    t = torch.tensor((TEMP[idx] - temp_mu) / temp_sd, dtype=torch.float32).unsqueeze(1)
    b = torch.tensor((BVP[idx] - bvp_mu) / bvp_sd, dtype=torch.float32).unsqueeze(1)
    a = torch.tensor((ACC[idx] - acc_mu) / acc_sd, dtype=torch.float32).permute(0, 2, 1)
    y = torch.tensor(label[idx], dtype=torch.float32)
    return e, t, b, a, y


def augment(x):
    """x: (batch, channels, length), already z-scored. Adds jitter + a small per-example
    circular time-shift, independently per example. Training-only."""
    x = x + torch.randn_like(x) * NOISE_STD
    length = x.shape[-1]
    max_shift = max(1, int(length * MAX_SHIFT_FRAC))
    shifts = torch.randint(-max_shift, max_shift + 1, (x.shape[0],))
    out = torch.empty_like(x)
    for i, s in enumerate(shifts.tolist()):
        out[i] = torch.roll(x[i], shifts=s, dims=-1)
    return out


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


def train_one_fold(train_idx, val_idx, EDA, TEMP, BVP, ACC, label):
    eda_mu, eda_sd = EDA[train_idx].mean(), EDA[train_idx].std() + 1e-8
    temp_mu, temp_sd = TEMP[train_idx].mean(), TEMP[train_idx].std() + 1e-8
    bvp_mu, bvp_sd = BVP[train_idx].mean(), BVP[train_idx].std() + 1e-8
    acc_mu, acc_sd = ACC[train_idx].mean(), ACC[train_idx].std() + 1e-8
    norm_args = (eda_mu, eda_sd, temp_mu, temp_sd, bvp_mu, bvp_sd, acc_mu, acc_sd)

    e_tr, t_tr, b_tr, a_tr, y_tr = to_tensors(train_idx, EDA, TEMP, BVP, ACC, label, *norm_args)
    e_va, t_va, b_va, a_va, y_va = to_tensors(val_idx, EDA, TEMP, BVP, ACC, label, *norm_args)

    model = WesadStressNet()
    n_pos, n_neg = y_tr.sum().item(), (1 - y_tr).sum().item()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    n = len(y_tr)
    best_val_auc, best_state, epochs_no_improve = -1.0, None, 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, BATCH_SIZE):
            batch_idx = perm[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            logits, _ = model(
                augment(e_tr[batch_idx]), augment(t_tr[batch_idx]),
                augment(b_tr[batch_idx]), augment(a_tr[batch_idx]),
            )
            loss = criterion(logits, y_tr[batch_idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits, _ = model(e_va, t_va, b_va, a_va)
            val_probs = torch.sigmoid(val_logits).numpy()
        val_auc = roc_auc_score(y_va.numpy(), val_probs) if len(np.unique(y_va.numpy())) > 1 else 0.5
        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            break

    model.load_state_dict(best_state)
    return model, norm_args, best_val_auc


def main():
    EDA, TEMP, BVP, ACC, label, subject = load_data()
    subjects = sorted(np.unique(subject), key=lambda s: int(s[1:]))
    print(f"Subjects: {len(subjects)} -> {subjects}")
    print(f"Windows: {len(label)}  Stress: {label.sum()} ({100*label.mean():.1f}%)")

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

        model, norm_args, best_val_auc = train_one_fold(train_idx, val_idx, EDA, TEMP, BVP, ACC, label)

        e_va, t_va, b_va, a_va, y_va = to_tensors(val_idx, EDA, TEMP, BVP, ACC, label, *norm_args)
        e_te, t_te, b_te, a_te, y_te = to_tensors(test_idx, EDA, TEMP, BVP, ACC, label, *norm_args)

        model.eval()
        with torch.no_grad():
            val_logits, _ = model(e_va, t_va, b_va, a_va)
            val_probs = torch.sigmoid(val_logits).numpy()
            test_logits, test_weights = model(e_te, t_te, b_te, a_te)
            test_probs = torch.sigmoid(test_logits).numpy()

        thr, val_f1 = best_f1_threshold(val_probs, y_va.numpy())
        y_te_np = y_te.numpy()
        test_auc = roc_auc_score(y_te_np, test_probs) if len(np.unique(y_te_np)) > 1 else float("nan")
        result = evaluate(test_probs, y_te_np, thr)
        result.update(subject=test_subj, val_subject=val_subj, threshold=thr,
                       val_f1=val_f1, val_auc=best_val_auc, test_auc=test_auc, n_test=len(y_te_np))
        fold_rows.append(result)
        pooled_probs.append(test_probs)
        pooled_y.append(y_te_np)
        all_weights.append(test_weights.detach().numpy())

        print(f"[{test_subj}] val_subj={val_subj} thr={thr:.3f} "
              f"test_auc={test_auc:.4f} test_f1={result['f1']:.4f} test_acc={result['acc']:.4f}")

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(FOLD_RESULTS_CSV, index=False)

    print("\n=== CNN+Attention v2 (regularized+augmented) LOSO-CV summary across 15 folds ===")
    print(f"AUC:       mean={fold_df['test_auc'].mean():.4f}  std={fold_df['test_auc'].std():.4f}")
    print(f"F1:        mean={fold_df['f1'].mean():.4f}  std={fold_df['f1'].std():.4f}")
    print(f"Accuracy:  mean={fold_df['acc'].mean():.4f}  std={fold_df['acc'].std():.4f}")
    print(f"Precision: mean={fold_df['precision'].mean():.4f}")
    print(f"Recall:    mean={fold_df['recall'].mean():.4f}")

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
    final_model, final_norm_args, _ = train_one_fold(train_idx, val_idx, EDA, TEMP, BVP, ACC, label)

    torch.save({
        "state_dict": final_model.state_dict(),
        "norm_args": final_norm_args,
        "threshold": pooled_thr,
    }, ROOT / "models" / "wesad_stress_cnn_attention_v2.pt")
    print(f"\nFinal model saved: models/wesad_stress_cnn_attention_v2.pt")


if __name__ == "__main__":
    main()
