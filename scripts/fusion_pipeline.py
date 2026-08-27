"""
Stage C: fuse Stage A (wearable stress) and Stage B (image severity) into one composite
score. This is DECISION-LEVEL (late) fusion, not a jointly-trained model -- deliberately,
because no dataset exists anywhere that has both wearable sensor data and skin images
from the same patients/sessions, so there is nothing to train a joint model on (see
the honors project plan discussion; same missing-data wall as the scratch-detection and
lesion-thermography ideas). This mirrors how real clinical severity indices like SCORAD/
EASI work: independently-assessed components combined by an explicit rule, not fit to a
giant labeled dataset.

Composite score = w_b * (Stage B image severity) + w_a * (Stage A stress score)

Both sub-scores are already probabilities in [0, 1] (sigmoid/softmax outputs), so no
further normalization is needed before combining. Weights default to equal (0.5/0.5) --
stated explicitly as a PROPOSED weighting, not one fit to data, since there's no ground
truth for the combined task to fit against.

This module exposes two independently-callable, independently-validated pieces:
  stage_b_predict(image_path)      -> Eczema probability, from the curated CNN
  stage_a_stress_predict(eda,temp,bvp,acc) -> stress probability, from the WESAD CNN
and a fuse() function that combines already-computed sub-scores.

Run this file directly for a small demo on real (but arbitrarily paired -- see caveat in
demo()) example data, to sanity-check the fused score behaves sensibly.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

ROOT = Path(r"C:\Users\tishy\Documents\Honors")

STAGE_B_MODEL_PATH = ROOT / "models" / "curated_resnet18_balanced.pt"
# v1, not the "fine-tuned" v2 -- v2's extra dropout+augmentation made LOSO-CV results
# WORSE (mean F1 0.628 -> 0.529), so the original, less-regularized model is the real
# best one. See docs/ writeup for the full comparison.
STAGE_A_MODEL_PATH = ROOT / "models" / "wesad_stress_cnn_attention.pt"
IMG_SIZE = 224
DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Stage B: image severity (Eczema probability)
# ---------------------------------------------------------------------------

_stage_b_model = None
_stage_b_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _load_stage_b():
    global _stage_b_model
    if _stage_b_model is None:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load(STAGE_B_MODEL_PATH, map_location=DEVICE))
        model.eval()
        _stage_b_model = model
    return _stage_b_model


def stage_b_predict(image_path):
    """Returns P(Eczema) in [0,1] for one image."""
    model = _load_stage_b()
    img = Image.open(image_path).convert("RGB")
    x = _stage_b_transform(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)
    return float(probs[0, 1])


# ---------------------------------------------------------------------------
# Stage A: wearable stress score, from the fine-tuned CNN+attention model
# ---------------------------------------------------------------------------

# Must match train_wesad_cnn_attention_v2.py's architecture exactly to load its weights.
EMBED_DIM = 32


class ModalityBranch(nn.Module):
    """Matches train_wesad_cnn_attention.py (v1) exactly -- no in-branch dropout,
    since v1 is the model actually being loaded here (see STAGE_A_MODEL_PATH note)."""
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


class WesadStressNet(nn.Module):
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


_stage_a_model = None
_stage_a_norm = None


def _load_stage_a():
    global _stage_a_model, _stage_a_norm
    if _stage_a_model is None:
        ckpt = torch.load(STAGE_A_MODEL_PATH, map_location=DEVICE, weights_only=False)
        model = WesadStressNet()
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        _stage_a_model = model
        _stage_a_norm = ckpt["norm_args"]  # (eda_mu, eda_sd, temp_mu, temp_sd, bvp_mu, bvp_sd, acc_mu, acc_sd)
    return _stage_a_model, _stage_a_norm


def stage_a_stress_predict(eda, temp, bvp, acc):
    """eda, temp: (120,) 4Hz 30s window. bvp: (1920,) 64Hz 30s window. acc: (960,3) 32Hz
    30s window. Returns P(stress) in [0,1]."""
    model, norm = _load_stage_a()
    eda_mu, eda_sd, temp_mu, temp_sd, bvp_mu, bvp_sd, acc_mu, acc_sd = norm

    e = torch.tensor((np.asarray(eda) - eda_mu) / eda_sd, dtype=torch.float32).view(1, 1, -1)
    t = torch.tensor((np.asarray(temp) - temp_mu) / temp_sd, dtype=torch.float32).view(1, 1, -1)
    b = torch.tensor((np.asarray(bvp) - bvp_mu) / bvp_sd, dtype=torch.float32).view(1, 1, -1)
    a = torch.tensor((np.asarray(acc) - acc_mu) / acc_sd, dtype=torch.float32).permute(1, 0).unsqueeze(0)

    with torch.no_grad():
        logit, weights = model(e, t, b, a)
        prob = torch.sigmoid(logit)
    return float(prob[0]), weights[0].numpy()


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def fuse(image_score, stress_score, w_b=0.5, w_a=0.5):
    """Proposed, literature-motivated weighting (SCORAD/EASI-style additive composite),
    NOT fit to data -- no dataset exists with both modalities on the same patients to fit
    weights against. Equal weighting by default; change w_b/w_a to explore sensitivity."""
    assert abs(w_b + w_a - 1.0) < 1e-6, "weights should sum to 1"
    return w_b * image_score + w_a * stress_score


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    """Runs the real Stage B model on real test images, and the real Stage A model on
    real held-out WESAD windows, then fuses each combination. IMPORTANT CAVEAT: the image
    and the wearable window are NOT from the same person or moment -- there is no dataset
    that provides that pairing (see module docstring). This demo only checks that the
    fusion architecture runs correctly and produces sensible, monotonic output given its
    two inputs; it is not a claim about any real patient's combined risk."""
    import csv

    test_manifest = ROOT / "SkinDisease" / "manifest_curated_v3_test.csv"
    with open(test_manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    eczema_img = next(r["path"] for r in rows if r["label"] == "1")
    other_img = next(r["path"] for r in rows if r["label"] == "0")

    npz = np.load(ROOT / "dataset" / "WESAD" / "wesad_raw_windows.npz", allow_pickle=True)
    subject = npz["subject"]
    label = npz["label"]
    # pick one real stress window and one real non-stress window (from the same subject,
    # arbitrarily S17, just so the two example windows are at least internally consistent)
    s = "S17"
    idx_stress = np.where((subject == s) & (label == 1))[0][0]
    idx_calm = np.where((subject == s) & (label == 0))[0][0]

    def stress_score(idx):
        eda = npz["EDA"][idx].squeeze(-1)
        temp = npz["TEMP"][idx].squeeze(-1)
        bvp = npz["BVP"][idx].squeeze(-1)
        acc = npz["ACC"][idx]
        prob, weights = stage_a_stress_predict(eda, temp, bvp, acc)
        return prob, weights

    img_score_eczema = stage_b_predict(eczema_img)
    img_score_other = stage_b_predict(other_img)
    stress_prob_high, w_high = stress_score(idx_stress)
    stress_prob_low, w_low = stress_score(idx_calm)

    print("Stage B (image) scores:")
    print(f"  Eczema image  -> P(Eczema) = {img_score_eczema:.3f}")
    print(f"  Other image   -> P(Eczema) = {img_score_other:.3f}")
    print(f"\nStage A (wearable) scores, subject {s}:")
    print(f"  Known-stress window     -> P(stress) = {stress_prob_high:.3f}  (attn EDA/TEMP/BVP/ACC = {w_high.round(2)})")
    print(f"  Known-not-stress window -> P(stress) = {stress_prob_low:.3f}  (attn EDA/TEMP/BVP/ACC = {w_low.round(2)})")

    print("\nFused composite scores (equal weights, illustrative pairings only -- see demo() docstring):")
    combos = [
        ("Eczema image + high stress", img_score_eczema, stress_prob_high),
        ("Eczema image + low stress", img_score_eczema, stress_prob_low),
        ("Other image + high stress", img_score_other, stress_prob_high),
        ("Other image + low stress", img_score_other, stress_prob_low),
    ]
    for name, img_s, stress_s in combos:
        print(f"  {name:32s} -> composite = {fuse(img_s, stress_s):.3f}")


if __name__ == "__main__":
    demo()
