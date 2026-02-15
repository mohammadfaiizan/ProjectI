"""
Scikit-learn TSNE: perplexity, n_components=2, learning_rate
"""

import numpy as np
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler


def main():
    print("=" * 60)
    print("t-SNE Visualization")
    print("=" * 60)

    X, y = load_digits(return_X_y=True)
    X = StandardScaler().fit_transform(X[:500])

    print("\n[1] TSNE basic (n_components=2):")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X)
    print(f"    Transformed shape: {X_tsne.shape}")
    print(f"    KL divergence: {tsne.kl_divergence_:.4f}")

    print("\n[2] perplexity effect:")
    for perp in [5, 30, 50]:
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
        X_t = tsne.fit_transform(X)
        print(f"    perplexity={perp}: range [{X_t.min():.2f}, {X_t.max():.2f}]")

    print("\n[3] learning_rate:")
    for lr in [100, 200, 500]:
        tsne = TSNE(n_components=2, learning_rate=lr, random_state=42)
        tsne.fit_transform(X)
        print(f"    learning_rate={lr}: n_iter={tsne.n_iter_}")

    print("\n[4] n_components=3 (rare):")
    tsne3 = TSNE(n_components=3, perplexity=30, random_state=42)
    X_3d = tsne3.fit_transform(X)
    print(f"    3D output shape: {X_3d.shape}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
