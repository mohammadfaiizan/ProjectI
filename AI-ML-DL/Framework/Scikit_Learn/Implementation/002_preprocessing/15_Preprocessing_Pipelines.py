"""
Scikit-learn: Combining preprocessors with Pipeline and ColumnTransformer
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

np.random.seed(42)
X_num = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
X_cat = np.array([['A', 'X'], ['B', 'Y'], ['A', 'Y'], ['B', 'X']])
X = np.hstack([X_num, X_cat])
print("Mixed data (numeric + categorical):")
print(X)

num_cols = [0, 1]
cat_cols = [2, 3]

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

X_transformed = preprocessor.fit_transform(X)
print("\nTransformed (Pipeline + ColumnTransformer):")
print(X_transformed)
print("Shape:", X_transformed.shape)

print("\n--- Full pipeline with estimator ---")
from sklearn.linear_model import LogisticRegression
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])
y = np.array([0, 1, 0, 1])
full_pipeline.fit(X, y)
print("Pipeline steps:", [s[0] for s in full_pipeline.steps])
print("Predict:", full_pipeline.predict(X))
