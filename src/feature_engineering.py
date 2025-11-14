from preprocessing import clean_text
from td_idf import TFIDFVectorizer

class TFIDFFeatureEngineer:
    def init(self, max_features=5000):
        self.max_features = max_features
        self.vectorizer = TFIDFVectorizer(max_features=max_features)
    def clean(self, texts):
        return [clean_text(t) for t in texts]
    def fit:
    def transform:
    def fit_transform: