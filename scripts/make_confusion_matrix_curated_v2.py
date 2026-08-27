import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

cm_v1 = np.array([[467, 208], [45, 111]])   # first CNN attempt (frozen backbone), 69.55%
cm_v2 = np.array([[631, 44], [64, 92]])     # fine-tuned CNN (layer4+fc unfrozen), 87.00%
labels = ["Other", "Eczema"]

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, cm, title in zip(axes, [cm_v1, cm_v2],
                          ["CNN v1 (frozen backbone)\ntest acc 69.55%",
                           "CNN v2 (fine-tuned layer4+fc)\ntest acc 87.00%"]):
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=13, fontweight="bold")

plt.suptitle("Eczema vs. Curated Similar-Looking Diseases — CNN Before/After Fine-Tuning (n=831)")
plt.tight_layout()
plt.savefig(r"C:\Users\tishy\Documents\Honors\docs\confusion_matrix_curated_v2.png", dpi=150)
print("saved")
