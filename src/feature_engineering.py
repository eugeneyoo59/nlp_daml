from utils.tf_idf import tf_idf_vectorize
from utils.preprocessing import preprocess

class TFIDFFeatureEngineer:
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.vectorizer = tf_idf_vectorize(max_features=max_features)
    def clean(self, texts):
        return [preprocess(t) for t in texts]
    def fit(self, texts):
        cleaned = self.clean(texts)
        return self.vectorizer.fit(cleaned)
    def transform(self, texts):
        cleaned = self.clean(texts)
        return self.vectorizer.transform(cleaned)
    def fit_transform(self, texts):
        cleaned = self.clean(texts)
        return self.vectorizer.fit_transform(cleaned)
    