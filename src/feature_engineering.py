# src/utils/feature_engineering.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from utils.preprocessing import preprocess


class TFIDFFeatureEngineer:
    """
    Wraps TF-IDF vectorization + preprocessing for our fake news dataset.

    - Cleans (title + text) with `preprocess`
    - Uses sklearn's TfidfVectorizer under the hood (fast, sparse)
    """

    def __init__(self, max_features=5000):
        """
        max_features: cap on vocabulary size (most frequent terms).
        You can tweak this (e.g. 2000, 10000) depending on speed/accuracy.
        """
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            preprocessor=None,    # we call preprocess() ourselves
            tokenizer=str.split,  # assume preprocess returns space-separated tokens
            lowercase=False       # preprocess should handle casing if needed
        )

    def clean(self, texts):
        """
        Apply your custom preprocess() to each text.
        `texts` is an iterable of raw strings (e.g., title + text).
        """
        return [preprocess(t) for t in texts]

    def fit(self, texts):
        """
        Fit TF-IDF vocabulary/model on training texts.
        """
        cleaned = self.clean(texts)
        return self.vectorizer.fit(cleaned)

    def transform(self, texts):
        """
        Transform texts to TF-IDF features using the fitted vectorizer.
        Returns a sparse matrix.
        """
        cleaned = self.clean(texts)
        return self.vectorizer.transform(cleaned)

    def fit_transform(self, texts):
        """
        Fit on texts, then return their TF-IDF features.
        """
        cleaned = self.clean(texts)
        return self.vectorizer.fit_transform(cleaned)

    @staticmethod
    def split_data(df, test_size=0.2, random_state=42, stratify=True):
        """
        Split dataframe into train/test:
        - X = title + text
        - y = label
        """
        # Combine title and text into a single string per row
        X = df["title"].astype(str) + " " + df["text"].astype(str)
        y = df["label"].values

        stratify_arg = y if stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_arg,
        )
        return X_train, X_test, y_train, y_test
