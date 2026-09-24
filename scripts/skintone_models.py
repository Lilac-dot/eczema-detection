"""
Starting models for the skin-tone audit, each returned with a single-logit eczema head so
all downstream code (BCEWithLogitsLoss, sigmoid probabilities) treats them the same way.

  resnet18            curated Stage B ResNet18 (models/curated_resnet18_balanced.pt,
                      train_curated_cnn_balanced.py: ImageNet ResNet18 with a
                      Linear(512, 2) softmax head, class index 1 = Eczema).
                      Its 2-way head is converted exactly to one logit:
                      softmax(z)[1] = sigmoid(z1 - z0), so the new Linear(512, 1) gets
                      weight w1 - w0 and bias b1 - b0 -- identical predictions.
  shufflenet_v2_x0_5, shufflenet_v2_x1_0, squeezenet1_1
                      curated checkpoints from the architecture comparison
                      (experiments/<name>/checkpoints/best_overall.pt), already
                      single-logit. These three plus ResNet18 are the architectures that
                      were backend-stable (or ResNet18: the reference) in the compression
                      study, papers/edge-ai-lightweight-deployment.

init="imagenet" returns the same architectures with ImageNet weights and a fresh head.

BLOCKS lists each model's backbone blocks from input to output, which the fine-tuning
scripts use to decide what to unfreeze.
"""
import torch
import torch.nn as nn
from torchvision import models

from paths import ROOT, MODELS_DIR

BLOCKS = {
    "resnet18": ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"],
    "shufflenet_v2_x0_5": ["conv1", "stage2", "stage3", "stage4", "conv5"],
    "shufflenet_v2_x1_0": ["conv1", "stage2", "stage3", "stage4", "conv5"],
    "squeezenet1_1": [f"features.{i}" for i in range(13)],
}
HEAD = {"resnet18": "fc", "shufflenet_v2_x0_5": "fc", "shufflenet_v2_x1_0": "fc",
        "squeezenet1_1": "classifier"}


def load_start_model(name, init="curated"):
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        if init == "curated":
            sd = torch.load(MODELS_DIR / "curated_resnet18_balanced.pt", map_location="cpu")
            w, b = sd.pop("fc.weight"), sd.pop("fc.bias")
            model.fc = nn.Identity()
            model.load_state_dict(sd)
            model.fc = nn.Linear(512, 1)
            with torch.no_grad():
                model.fc.weight.copy_((w[1] - w[0]).unsqueeze(0))
                model.fc.bias.copy_((b[1] - b[0]).unsqueeze(0))
        else:
            model.fc = nn.Linear(512, 1)
        return model
    if name in ("shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "squeezenet1_1"):
        from train_transfer_cnn import build_model
        model = build_model(name)
        if init == "curated":
            ckpt = ROOT / "experiments" / name / "checkpoints" / "best_overall.pt"
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        for p in model.parameters():
            p.requires_grad = True
        return model
    raise ValueError(name)


def make_skintone_transforms(img_size=224):
    """Aspect-preserving versions of train_transfer_cnn.make_transforms: resize the shorter
    side, then crop, instead of squashing to a square. Colour augmentation is limited to
    brightness/contrast -- no hue/saturation jitter, which would blur the very skin-tone
    differences this audit measures."""
    from torchvision import transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    resize = int(img_size * 256 / 224)
    train_tf = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(3 / 4, 4 / 3)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        normalize,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf
