import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from utils.tf_idf import tf_idf_vectorize
from utils.text2vec import vectorize

tf_idf_knn = Pipeline([
    ("norm", Normalizer(norm="l2")), # normalize tf-idf vectors
    ("knn", KNeighborsClassifier(algorithm="brute", metric="cosine", weights="distance"))
])
