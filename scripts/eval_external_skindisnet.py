"""Run the deployed Stage B model (unmodified) against the SkinDisNet external manifest."""
from paths import SKINDISNET_DIR
from eval_external_common import run_eval

if __name__ == "__main__":
    run_eval(SKINDISNET_DIR / "manifest_skindisnet.csv", "SkinDisNet (external)")
