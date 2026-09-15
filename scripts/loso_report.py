"""Shared LOSO-CV reliability reporting, used by both train_wesad_stress.py (LightGBM)
and train_wesad_cnn_attention.py (CNN+attention).

A mean AUC across folds can look strong while hiding folds where the decision threshold
(tuned on a held-out VALIDATION subject and applied to a different TEST subject) fails to
transfer -- e.g. a fold can have test_auc=1.0 (the model ranks stress windows above
non-stress windows perfectly) yet f1=0.0 (the specific threshold carried over from another
subject predicts everyone the same class for this one). Reporting only the mean AUC hides
this; per-subject breakdown does not.
"""


def print_reliability_summary(fold_df, low_f1_threshold=0.3):
    print("\n=== Per-subject reliability (sorted worst-F1-first) ===")
    print(f"{'subject':>8} {'test_auc':>9} {'f1':>7} {'acc':>7} {'threshold':>10}")
    for _, row in fold_df.sort_values("f1").iterrows():
        print(f"{row['subject']:>8} {row['test_auc']:9.4f} {row['f1']:7.4f} "
              f"{row['acc']:7.4f} {row['threshold']:10.4f}")

    n_low_f1 = int((fold_df["f1"] < low_f1_threshold).sum())
    n_total = len(fold_df)
    thr_min, thr_max = fold_df["threshold"].min(), fold_df["threshold"].max()
    auc_mean = fold_df["test_auc"].mean()

    print(f"\nAUC range:       [{fold_df['test_auc'].min():.4f}, {fold_df['test_auc'].max():.4f}]")
    print(f"F1 range:        [{fold_df['f1'].min():.4f}, {fold_df['f1'].max():.4f}]")
    print(f"Threshold range: [{thr_min:.4f}, {thr_max:.4f}]  (per-fold, val-subject-tuned)")
    print(f"\n{n_low_f1}/{n_total} subjects have F1 < {low_f1_threshold} despite a mean AUC of "
          f"{auc_mean:.4f}.")
    print(
        "A high mean AUC means the model's raw scores rank stress above non-stress "
        "reasonably well on average -- it does NOT mean a single fixed decision threshold "
        "is safe to deploy for a new person. The threshold range above shows the val-tuned "
        "cutoff swings widely fold to fold; several test subjects get near-total false "
        "positives or false negatives once that cutoff is applied to them. Treat the mean "
        "AUC as a ceiling on what per-subject recalibration could achieve, not as the "
        "expected out-of-the-box performance for an unseen user."
    )
    return dict(n_low_f1=n_low_f1, n_total=n_total, threshold_min=float(thr_min),
                threshold_max=float(thr_max))
