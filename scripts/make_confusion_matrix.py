import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

cm = np.array([[183, 17], [1, 214]])  # rows: true Normal/Eczema; cols: pred Normal/Eczema
labels = ["Normal", "Eczema"]

fig, ax = plt.subplots(figsize=(4.5, 4))
im = ax.imshow(cm, cmap="Blues")

ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("Stage B Test Set Confusion Matrix (n=415)")

for i in range(2):
    for j in range(2):
        val = cm[i, j]
        color = "white" if val > cm.max() / 2 else "black"
        ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=14, fontweight="bold")

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(r"C:\Users\tishy\Documents\Honors\docs\stage_b_original_cnn_confusion_matrix.png", dpi=150)
print("saved")
