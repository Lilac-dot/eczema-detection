"""Quantify domain shift between Internal / SCIN / SkinDisNet using the deployed CNN's
own embedding space: pairwise proxy A-distance (Ben-David et al. 2010), a 3-way origin
classifier, and a PCA visualization. Ties back to docs/external_generalization_
improvement_2026-09-15.md's finding that multi-source fine-tuning helped SkinDisNet's
AUC substantially but SCIN's barely at all -- the question this script answers is
whether that asymmetry is predicted by how separable each external dataset is from
Internal in feature space.

Reads scripts/extract_domain_embeddings.py's cached output. No training-set/test-set
leakage concern here: this classifier's "labels" are dataset ORIGIN, not the eczema
task label, and its own train/test split is disjoint -- it has nothing to do with the
Stage B model's own train/val/test split.
"""
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from paths import ROOT

EMBEDDINGS_PATH = ROOT / "dataset" / "embeddings_internal_scin_skindisnet.npz"
SEED = 42
N_BOOTSTRAP = 1000

PARAMS_BINARY = dict(objective="binary", metric="binary_error", verbosity=-1, seed=SEED,
                      learning_rate=0.05, num_leaves=15)
PARAMS_MULTI = dict(objective="multiclass", num_class=3, metric="multi_error", verbosity=-1,
                     seed=SEED, learning_rate=0.05, num_leaves=15)


def load_data():
    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    return data["embeddings"], data["sources"], data["paths"]


def bootstrap_ci(correct, metric_fn, n=N_BOOTSTRAP, seed=SEED):
    """correct: boolean array, one entry per test sample, indicating whether the
    classifier got that sample right. Bootstraps over test SAMPLES (images), matching
    eval_external_common.py's existing bootstrap pattern."""
    rng = np.random.RandomState(seed)
    n_samples = len(correct)
    vals = []
    for _ in range(n):
        idx = rng.randint(0, n_samples, n_samples)
        vals.append(metric_fn(correct[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def err_to_a_distance(err):
    return 2 * (1 - 2 * err)


def pairwise_a_distance(embeddings, sources, source_a, source_b):
    mask = (sources == source_a) | (sources == source_b)
    X = embeddings[mask]
    y = (sources[mask] == source_a).astype(int)  # arbitrary but fixed convention

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED)

    train_set = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(PARAMS_BINARY, train_set, num_boost_round=100)

    probs = model.predict(X_test)
    preds = (probs >= 0.5).astype(int)
    correct = (preds == y_test)
    acc = correct.mean()
    err = 1 - acc
    a_dist = err_to_a_distance(err)

    def a_dist_metric(correct_resampled):
        err_r = 1 - correct_resampled.mean()
        return err_to_a_distance(err_r)

    ci_lo, ci_hi = bootstrap_ci(correct, a_dist_metric)

    print(f"\n{source_a} vs {source_b}: n_train={len(X_train)} n_test={len(X_test)} "
          f"classifier_test_acc={acc:.4f} err={err:.4f} "
          f"proxy_A_distance={a_dist:.4f} (95% CI {ci_lo:.4f}-{ci_hi:.4f})")

    return dict(source_a=source_a, source_b=source_b, n_train=len(X_train), n_test=len(X_test),
                classifier_test_acc=float(acc), err=float(err), a_distance=float(a_dist),
                a_distance_ci=(ci_lo, ci_hi))


def three_way_classifier(embeddings, sources):
    label_names = sorted(np.unique(sources))
    label_map = {name: i for i, name in enumerate(label_names)}
    y = np.array([label_map[s] for s in sources])

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y, test_size=0.3, stratify=y, random_state=SEED)

    train_set = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(PARAMS_MULTI, train_set, num_boost_round=150)

    probs = model.predict(X_test)
    preds = probs.argmax(axis=1)
    correct = (preds == y_test)
    acc = correct.mean()

    def acc_metric(correct_resampled):
        return correct_resampled.mean()

    ci_lo, ci_hi = bootstrap_ci(correct, acc_metric)

    cm = confusion_matrix(y_test, preds, labels=list(range(len(label_names))))

    print(f"\n3-way origin classifier: n_train={len(X_train)} n_test={len(X_test)} "
          f"accuracy={acc:.4f} (95% CI {ci_lo:.4f}-{ci_hi:.4f})  "
          f"(chance level ~{1/len(label_names):.4f})")
    print(f"Labels (row=true, col=predicted): {label_names}")
    print(cm)

    return dict(label_names=label_names, accuracy=float(acc), accuracy_ci=(ci_lo, ci_hi),
                confusion_matrix=cm.tolist(), n_train=len(X_train), n_test=len(X_test))


def make_pca_plot(embeddings, sources, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"Internal": "#1f77b4", "SCIN": "#ff7f0e", "SkinDisNet": "#2ca02c"}
    for source in sorted(np.unique(sources)):
        mask = sources == source
        ax.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.4,
                   label=f"{source} (n={mask.sum()})", c=colors.get(source, "gray"))
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("PCA of deployed ResNet18's penultimate-layer embeddings\nby dataset origin")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved PCA plot: {out_path}")
    return dict(explained_variance_ratio=pca.explained_variance_ratio_[:2].tolist())


def main():
    embeddings, sources, paths = load_data()
    print(f"Loaded {len(embeddings)} embeddings, sources: {dict(zip(*np.unique(sources, return_counts=True)))}")

    pair_results = []
    for a, b in [("Internal", "SCIN"), ("Internal", "SkinDisNet"), ("SCIN", "SkinDisNet")]:
        pair_results.append(pairwise_a_distance(embeddings, sources, a, b))

    multi_result = three_way_classifier(embeddings, sources)

    pca_out = ROOT / "docs" / "domain_shift_pca_2026-09-15.png"
    pca_result = make_pca_plot(embeddings, sources, pca_out)

    return dict(pairwise=pair_results, three_way=multi_result, pca=pca_result)


if __name__ == "__main__":
    main()
