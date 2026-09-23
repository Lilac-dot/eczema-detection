"""
Small leave-one-subject-out (LOSO) model on the Southampton e-textile capacitive
moisture sensor data: lesional vs. non-lesional AD skin, from raw sensor capacitance
alone (no clinical-instrument readings used as input -- Corneometer/TEWL are the
validation ground truth this sensor is being compared against, not features a deployed
device would have access to).

N = 13 patients, 2 recordings each (lesional/non-lesional) = 26 total rows. This is far
too small for a train/test split to mean anything, so every prediction here is made on a
patient the model never saw during that fold's training -- same protocol already used for
WESAD elsewhere in this project. No literal "transfer learning" is done: there is no
pretrained backbone for capacitance sensor data to transfer from. This trains directly on
the 26 rows, with LOSO as the only defensible validation given the sample size.

Data source: dataset/southampton_moisture/PatientTest/ (see SOURCE.md).
"""
import csv

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from scipy.stats import wilcoxon

from paths import ROOT

DATA_DIR = ROOT / "dataset" / "southampton_moisture" / "PatientTest"


def load_timeseries(fname):
    """Returns dict: patient_id -> list of float readings (time-ordered)."""
    with open(DATA_DIR / fname, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        patients = header[1:]
        series = {p: [] for p in patients}
        for row in reader:
            for p, v in zip(patients, row[1:]):
                series[p].append(float(v))
    return series


def extract_features(values):
    values = np.array(values, dtype=float)
    t = np.arange(len(values))
    slope = np.polyfit(t, values, 1)[0]
    return {
        "mean": values.mean(),
        "std": values.std(),
        "slope": slope,
        "first": values[0],
        "last": values[-1],
        "range": values.max() - values.min(),
    }


def main():
    lesional = load_timeseries("lesional_sensor.csv")
    nonlesional = load_timeseries("nonlesional_sensor.csv")
    patients = sorted(lesional.keys())
    assert patients == sorted(nonlesional.keys()), "patient columns must match between files"

    feature_names = ["mean", "std", "slope", "first", "last", "range"]
    rows = []  # (patient, label, feature_vector)
    for p in patients:
        f_les = extract_features(lesional[p])
        f_non = extract_features(nonlesional[p])
        rows.append((p, 1, [f_les[k] for k in feature_names]))
        rows.append((p, 0, [f_non[k] for k in feature_names]))

    X = np.array([r[2] for r in rows])
    y = np.array([r[1] for r in rows])
    groups = np.array([r[0] for r in rows])

    print(f"N = {len(patients)} patients, {len(rows)} total recordings "
          f"({sum(y)} lesional, {len(y) - sum(y)} non-lesional)")
    print(f"Features: {feature_names}\n")

    print("=== Sanity check: paired comparison on mean capacitance alone ===")
    les_means = np.array([extract_features(lesional[p])["mean"] for p in patients])
    non_means = np.array([extract_features(nonlesional[p])["mean"] for p in patients])
    diff = non_means - les_means
    print(f"Non-lesional - lesional mean capacitance per patient: "
          f"mean diff = {diff.mean():.2f} pF, sd = {diff.std():.2f} pF")
    print(f"Direction: non-lesional higher in {int((diff > 0).sum())}/{len(patients)} patients")
    stat, p_val = wilcoxon(non_means, les_means)
    print(f"Wilcoxon signed-rank test (paired, n={len(patients)}): "
          f"statistic={stat:.1f}, p={p_val:.4f}\n")

    print("=== LOSO logistic regression (sensor-derived features only) ===")
    unique_patients = patients
    y_true_all, y_prob_all, y_pred_all = [], [], []
    fold_log = []
    for held_out in unique_patients:
        train_mask = groups != held_out
        test_mask = groups == held_out
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        mu, sigma = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
        X_train_n = (X_train - mu) / sigma
        X_test_n = (X_test - mu) / sigma

        clf = LogisticRegression(C=0.5, max_iter=1000)
        clf.fit(X_train_n, y_train)
        probs = clf.predict_proba(X_test_n)[:, 1]
        preds = (probs >= 0.5).astype(int)

        y_true_all.extend(y_test.tolist())
        y_prob_all.extend(probs.tolist())
        y_pred_all.extend(preds.tolist())
        fold_log.append((held_out, y_test.tolist(), preds.tolist(), probs.round(3).tolist()))

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_prob_all = np.array(y_prob_all)

    acc = (y_true_all == y_pred_all).mean()
    auc = roc_auc_score(y_true_all, y_prob_all)
    cm = confusion_matrix(y_true_all, y_pred_all)

    print(f"{'Patient':<10}{'True':<18}{'Pred':<18}{'P(lesional)'}")
    for held_out, yt, yp, prob in fold_log:
        print(f"{held_out:<10}{str(yt):<18}{str(yp):<18}{prob}")

    print(f"\nPooled LOSO accuracy: {acc:.3f} ({int((y_true_all == y_pred_all).sum())}/{len(y_true_all)})")
    print(f"Pooled LOSO ROC-AUC: {auc:.3f}")
    print(f"Confusion matrix [rows=true(0,1), cols=pred(0,1)]:\n{cm}")

    # 95% CI on accuracy via simple binomial (Wilson score), honest about n
    from statsmodels.stats.proportion import proportion_confint
    lo, hi = proportion_confint(int((y_true_all == y_pred_all).sum()), len(y_true_all), method="wilson")
    print(f"95% Wilson CI on accuracy: [{lo:.3f}, {hi:.3f}] (n={len(y_true_all)} predictions, "
          f"only {len(unique_patients)} independent subjects)")


if __name__ == "__main__":
    main()
