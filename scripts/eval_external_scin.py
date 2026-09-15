"""Run the deployed Stage B model (unmodified) against the SCIN external manifest."""
from paths import SCIN_DIR
from eval_external_common import run_eval

if __name__ == "__main__":
    run_eval(SCIN_DIR / "manifest_scin.csv", "SCIN (external)")
