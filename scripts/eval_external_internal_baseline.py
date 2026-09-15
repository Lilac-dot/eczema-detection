"""Re-run the deployed model on its OWN internal test split through eval_external_common's
AUC/bootstrap-CI code path, so internal and external numbers are directly comparable
(the original eval_curated_cnn_balanced.py never computed AUC or a confidence interval)."""
from paths import SKINDISEASE_DIR
from eval_external_common import run_eval

if __name__ == "__main__":
    run_eval(SKINDISEASE_DIR / "manifest_curated_v3_test.csv", "Internal test set (baseline)")
