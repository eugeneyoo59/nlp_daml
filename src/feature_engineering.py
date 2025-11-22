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

from sklearn.model_selection import train_test_split

def split_data(df, test_size=0.2, random_state=42, stratify=True):
    # Combine title and text into a single string per row
    X = (df['title'].astype(str) + " " + df['text'].astype(str))
    y = df['label'].values

    stratify_arg = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg
    )
    return X_train, X_test, y_train, y_test
