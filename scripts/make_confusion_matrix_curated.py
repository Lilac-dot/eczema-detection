import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# CNN result
cm_cnn = np.array([[467, 208], [45, 111]])
# LightGBM result
cm_lgbm = np.array([[588, 87], [59, 97]])
labels = ["Other", "Eczema"]

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, cm, title in zip(axes, [cm_cnn, cm_lgbm],
                          ["ResNet18 CNN\n(test acc 69.55%)", "Colour-Feature + LightGBM\n(test acc 82.43%)"]):
    im = ax.imshow(cm, cmap="Purples")
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

plt.suptitle("Eczema vs. Curated Similar-Looking Diseases — Test Set (n=831)")
plt.tight_layout()
plt.savefig(r"C:\Users\tishy\Documents\Honors\docs\confusion_matrix_curated.png", dpi=150)
print("saved")
