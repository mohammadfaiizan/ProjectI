"""
Visualizing search results: heatmaps, parallel coordinates
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

try:
    import matplotlib.pyplot as plt
    import pandas as pd
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def main():
    print("=" * 60)
    print("Visualizing search results: heatmaps, parallel coordinates")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        "C": [0.1, 1.0, 10.0],
        "gamma": ["scale", 0.01, 0.1],
        "kernel": ["rbf"],
    }
    grid = GridSearchCV(SVC(random_state=42), param_grid, cv=3)
    grid.fit(X_train, y_train)

    print("\n[1] cv_results_ as DataFrame for analysis:")
    results = pd.DataFrame(grid.cv_results_)
    print(f"    Columns: {list(results.columns)[:8]}...")
    print(f"    Rows: {len(results)}")

    print("\n[2] Heatmap data (C vs gamma for kernel='rbf'):")
    subset = results[results["param_kernel"] == "rbf"]
    pivot = subset.pivot_table(
        values="mean_test_score",
        index="param_gamma",
        columns="param_C",
    )
    print(pivot.to_string())

    if MATPLOTLIB_AVAILABLE:
        print("\n[3] Creating heatmap and saving to file:")
        import os
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(x) for x in pivot.index])
        ax.set_xlabel("C")
        ax.set_ylabel("gamma")
        plt.colorbar(im, ax=ax, label="mean_test_score")
        plt.tight_layout()
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heatmap_example.png")
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"    Saved: heatmap_example.png")
    else:
        print("\n[3] matplotlib not available, skipping plot")

    print("\n[4] Parallel coordinates concept:")
    print("    Each row = one param combo, each column = param or score")
    print("    Use pd.plotting.parallel_coordinates or plotly for interactive")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
