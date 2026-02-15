"""
Scikit-learn permutation_importance (n_repeats, scoring)
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

np.random.seed(42)
X, y = make_classification(n_samples=300, n_features=10, n_informative=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train, y_train)

print("--- permutation_importance on test set ---")
result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
print("Importances (mean):", np.round(result.importances_mean, 4))
print("Importances (std):", np.round(result.importances_std, 4))
print("Top 3 features:", np.argsort(result.importances_mean)[-3:][::-1])

print("\n--- With custom scoring ---")
result_acc = permutation_importance(
    rf, X_test, y_test,
    n_repeats=10,
    scoring='accuracy',
    random_state=42
)
print("Accuracy-based importances:", np.round(result_acc.importances_mean, 4))

print("\n--- On training set (overfitting check) ---")
result_train = permutation_importance(
    rf, X_train, y_train,
    n_repeats=5,
    random_state=42
)
print("Train importances:", np.round(result_train.importances_mean, 4))
print("Test importances:", np.round(result.importances_mean, 4))
