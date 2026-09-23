"""
Patient-wise train/test comparison of pretrained tabular models on the Southampton
moisture sensor data (lesional vs. non-lesional AD skin, from raw sensor capacitance
features only -- no Corneometer/TEWL used as input, since a deployed device wouldn't
have those).

TEST-SET ISOLATION: 13 patients are split ONCE, by patient (never by row -- both a
patient's lesional and non-lesional recordings always land in the same split), into 9
train / 4 test, fixed seed. The 4 test patients' data is not used for anything except
the single final evaluation at the end of this script -- no model selection, no
threshold tuning, no peeking. This mirrors the test-isolation protocol already used for
the image-model comparison (scripts/eval_transfer_cnn_finalists.py).

Models compared, all trained/fit on the SAME 9-patient training set, evaluated on the
SAME held-out 4-patient test set:
  1. TabPFN v2      -- pretrained tabular foundation model, in-context learning
  2. TabICL         -- pretrained tabular foundation model, in-context learning
  3. Logistic Regression -- classical baseline, trained from scratch on the 18 train rows
  4. LightGBM       -- classical baseline, trained from scratch (already used elsewhere
                        in this project, e.g. the deployed WESAD stress model)

Given n=13 patients total (9 train / 4 test = 8 test recordings), every number here
carries a very wide confidence interval. This script reports raw counts and metrics
honestly -- it does not attempt to hide how small the test set is.
"""
import csv

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, f1_score
import lightgbm as lgb

from paths import ROOT

DATA_DIR = ROOT / "dataset" / "southampton_moisture" / "PatientTest"
SEED = 42
N_TEST_PATIENTS = 4
FEATURE_NAMES = ["mean", "std", "slope", "first", "last", "range"]


def load_timeseries(fname):
    """Some patients' recordings stop before the full 30s (real session-length variation,
    not a data error) -- their later rows have empty cells. Each patient's series is
    truncated to however many samples they actually have, rather than assuming a fixed
    length for everyone."""
    with open(DATA_DIR / fname, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        patients = header[1:]
        series = {p: [] for p in patients}
        for row in reader:
            for p, v in zip(patients, row[1:]):
                if v.strip() != "":
                    series[p].append(float(v))
    for p, vals in series.items():
        assert len(vals) >= 5, f"{fname}: patient {p} has only {len(vals)} samples -- too few to trust"
    return series


def extract_features(values):
    values = np.array(values, dtype=float)
    t = np.arange(len(values))
    slope = np.polyfit(t, values, 1)[0]
    return [values.mean(), values.std(), slope, values[0], values[-1],
            values.max() - values.min()]


def build_dataset():
    lesional = load_timeseries("lesional_sensor.csv")
    nonlesional = load_timeseries("nonlesional_sensor.csv")
    patients = sorted(lesional.keys())
    assert patients == sorted(nonlesional.keys())

    rows = []
    for p in patients:
        rows.append((p, 1, extract_features(lesional[p])))
        rows.append((p, 0, extract_features(nonlesional[p])))
    return patients, rows


def patient_split(patients, n_test, seed):
    rng = np.random.RandomState(seed)
    shuffled = list(patients)
    rng.shuffle(shuffled)
    test_patients = set(shuffled[:n_test])
    train_patients = set(shuffled[n_test:])
    return train_patients, test_patients


def to_xy(rows, patient_set):
    sub = [r for r in rows if r[0] in patient_set]
    X = np.array([r[2] for r in sub])
    y = np.array([r[1] for r in sub])
    pids = [r[0] for r in sub]
    return X, y, pids


def evaluate(name, y_test, probs):
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, probs)
    except ValueError:
        auc = float("nan")
    f1 = f1_score(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    print(f"\n--- {name} ---")
    print(f"Test predictions (prob of lesional): {np.round(probs, 3).tolist()}")
    print(f"True labels:                         {y_test.tolist()}")
    print(f"Accuracy: {acc:.3f} ({int((preds == y_test).sum())}/{len(y_test)})  "
          f"AUC: {auc:.3f}  F1: {f1:.3f}")
    print(f"Confusion matrix [rows=true(0,1), cols=pred(0,1)]:\n{cm}")
    return dict(name=name, acc=acc, auc=auc, f1=f1)


def main():
    patients, rows = build_dataset()
    train_patients, test_patients = patient_split(patients, N_TEST_PATIENTS, SEED)

    print(f"Total patients: {len(patients)}")
    print(f"Train patients ({len(train_patients)}): {sorted(train_patients)}")
    print(f"Test patients ({len(test_patients)}) -- HELD OUT, touched only at final eval: "
          f"{sorted(test_patients)}")

    X_train, y_train, _ = to_xy(rows, train_patients)
    X_test, y_test, test_pids = to_xy(rows, test_patients)
    print(f"Train rows: {len(y_train)}  Test rows: {len(y_test)}\n")

    mu, sigma = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
    X_train_n = (X_train - mu) / sigma
    X_test_n = (X_test - mu) / sigma

    results = []

    # --- 1. TabPFN v2 ---
    try:
        from tabpfn import TabPFNClassifier
        clf = TabPFNClassifier(random_state=SEED)
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        results.append(evaluate("TabPFN v2 (pretrained, in-context)", y_test, probs))
    except Exception as e:
        print(f"\n--- TabPFN v2: FAILED ({type(e).__name__}: {e}) ---")

    # --- 2. TabICL ---
    try:
        from tabicl import TabICLClassifier
        clf = TabICLClassifier(random_state=SEED)
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        results.append(evaluate("TabICL (pretrained, in-context)", y_test, probs))
    except Exception as e:
        print(f"\n--- TabICL: FAILED ({type(e).__name__}: {e}) ---")

    # --- 3. Logistic Regression (classical baseline, trained from scratch) ---
    clf = LogisticRegression(C=0.5, max_iter=1000, random_state=SEED)
    clf.fit(X_train_n, y_train)
    probs = clf.predict_proba(X_test_n)[:, 1]
    results.append(evaluate("Logistic Regression (from scratch)", y_test, probs))

    # --- 4. LightGBM (classical baseline, trained from scratch) ---
    clf = lgb.LGBMClassifier(n_estimators=50, max_depth=2, min_child_samples=2,
                              random_state=SEED, verbosity=-1)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    results.append(evaluate("LightGBM (from scratch)", y_test, probs))

    print("\n\n=== SUMMARY (test set: 4 held-out patients, 8 recordings) ===")
    print(f"{'Model':<40}{'Acc':<10}{'AUC':<10}{'F1'}")
    for r in results:
        print(f"{r['name']:<40}{r['acc']:<10.3f}{r['auc']:<10.3f}{r['f1']:.3f}")
    print("\nn=8 test predictions from 4 independent patients -- treat every number above "
          "as illustrative, not a validated result. See docs for the honest writeup.")


if __name__ == "__main__":
    main()
