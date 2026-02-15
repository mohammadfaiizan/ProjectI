"""
Scikit-learn GaussianMixture and BayesianGaussianMixture: n_components, covariance_type, BIC/AIC
"""

import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.datasets import make_blobs


def main():
    print("=" * 60)
    print("Gaussian Mixture Models")
    print("=" * 60)

    X, _ = make_blobs(n_samples=500, n_features=2, centers=4, random_state=42)

    print("\n[1] GaussianMixture basic usage:")
    gmm = GaussianMixture(n_components=4, covariance_type="full", random_state=42)
    gmm.fit(X)
    labels = gmm.predict(X)
    print(f"    labels_ (first 10): {labels[:10]}")
    print(f"    means_ shape: {gmm.means_.shape}")
    print(f"    converged: {gmm.converged_}")

    print("\n[2] BIC and AIC for model selection:")
    for k in range(2, 8):
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
        gmm.fit(X)
        print(f"    k={k}: BIC={gmm.bic(X):.2f}, AIC={gmm.aic(X):.2f}")

    print("\n[3] covariance_type options:")
    for cov in ["full", "tied", "diag"]:
        gmm = GaussianMixture(n_components=4, covariance_type=cov, random_state=42)
        gmm.fit(X)
        print(f"    covariance_type='{cov}': log-likelihood={gmm.score(X):.2f}")

    print("\n[4] BayesianGaussianMixture (auto n_components):")
    bgmm = BayesianGaussianMixture(n_components=10, weight_concentration_prior=0.1, random_state=42)
    bgmm.fit(X)
    active = np.sum(bgmm.weights_ > 0.01)
    print(f"    Active components (weight>0.01): {active}")
    print(f"    weights_: {bgmm.weights_.round(3)}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
