"""
Scikit-learn Complete feature engineering pipeline
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

np.random.seed(42)
X, y = make_classification(n_samples=300, n_features=20, n_informative=6, random_state=42)

print("--- Full feature engineering pipeline ---")
pipeline = Pipeline([
    ('variance', VarianceThreshold(threshold=0.01)),
    ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ('selector', SelectKBest(score_func=f_classif, k=15)),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])
scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print("CV accuracy:", np.round(scores, 3))
print("Mean CV accuracy:", np.round(scores.mean(), 3))

print("\n--- Pipeline steps ---")
for name, step in pipeline.steps:
    print(f"  {name}: {type(step).__name__}")

print("\n--- Fit and inspect ---")
pipeline.fit(X, y)
print("After variance - n_features:", pipeline.named_steps['variance'].get_support().sum())
print("After poly - n_features:", pipeline.named_steps['poly'].n_output_features_)
print("After selector - selected:", pipeline.named_steps['selector'].get_support().sum())

print("\n--- Simplified pipeline (no poly) ---")
pipe_simple = Pipeline([
    ('var', VarianceThreshold(threshold=0.0)),
    ('sel', SelectKBest(f_classif, k=10)),
    ('scale', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])
scores_simple = cross_val_score(pipe_simple, X, y, cv=5, scoring='accuracy')
print("Simple pipeline CV accuracy:", np.round(scores_simple.mean(), 3))
