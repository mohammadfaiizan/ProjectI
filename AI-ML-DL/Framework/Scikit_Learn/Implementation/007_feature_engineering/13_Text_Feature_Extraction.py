"""
Scikit-learn CountVectorizer, TfidfVectorizer, TfidfTransformer
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

docs = [
    "machine learning is great",
    "deep learning is powerful",
    "machine learning and deep learning",
]

print("--- CountVectorizer ---")
cv = CountVectorizer()
X_count = cv.fit_transform(docs)
print("Vocabulary:", cv.vocabulary_)
print("Shape:", X_count.toarray().shape)
print("Count matrix:\n", X_count.toarray())

print("\n--- TfidfVectorizer ---")
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(docs)
print("Vocabulary:", tfidf.vocabulary_)
print("TF-IDF matrix (rounded):\n", np.round(X_tfidf.toarray(), 3))

print("\n--- TfidfTransformer (on count matrix) ---")
tfidf_trans = TfidfTransformer()
X_tfidf2 = tfidf_trans.fit_transform(X_count)
print("TF-IDF from counts (rounded):\n", np.round(X_tfidf2.toarray(), 3))

print("\n--- max_features and ngram_range ---")
cv_ngram = CountVectorizer(ngram_range=(1, 2), max_features=10)
X_ngram = cv_ngram.fit_transform(docs)
print("Bigram vocabulary:", list(cv_ngram.vocabulary_.keys())[:10])
