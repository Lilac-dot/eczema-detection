"""Free-RAM checking, shared by any training script that needs to avoid an unclean OS
OOM kill. First used ad hoc in train_curated_cnn_multisource_weighted_v2.py
(docs/finetuning_v2_expanded_2026-09-16.md); pulled out here so later scripts (e.g.
train_lodo_resnet18.py) don't duplicate the same logic. Originally implemented via the
Windows-only ctypes GlobalMemoryStatusEx API (this project's earlier dev machine was
Windows); switched to psutil (already a project dependency, see
edge_ai_extended_analysis.py) so it also runs on macOS/Linux.
"""
import psutil


def free_mem_mb():
    """Free (available) physical RAM in MB, cross-platform via psutil."""
    return psutil.virtual_memory().available / (1024 * 1024)


def require_free_mb(min_mb, context=""):
    """Returns True if there's enough free RAM to proceed; prints and returns False
    (never raises) otherwise, so callers can abort a training loop gracefully instead of
    risking an unclean OS OOM kill mid-epoch."""
    free = free_mem_mb()
    if free < min_mb:
        label = f" ({context})" if context else ""
        print(f"Free RAM{label}: {free:.0f} MB, below the {min_mb:.0f} MB safety floor "
              f"-- not proceeding.")
        return False
    return True
