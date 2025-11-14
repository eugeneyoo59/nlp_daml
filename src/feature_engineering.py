from preprocessing import clean_text
from td_idf import TFIDFVectorizer

class TFIDFFeatureEngineer:
    def init(self, max_features=5000):
        self.max_features = max_features
        self.vectorizer = TFIDFVectorizer(max_features=max_features)
    def clean(self, texts):
        return [clean_text(t) for t in texts]
    def fit(self, texts):
        cleaned = self.clean(texts)
        return self.vectorizer.fit(cleaned)
    def transform(self, texts):
        cleaned = self.clean(texts)
        return self.vectorizer.transform(cleaned)
    def fit_transform(self, texts):
        cleaned = self.clean(texts)
        return self.vectorizer.fit_transform(cleaned)