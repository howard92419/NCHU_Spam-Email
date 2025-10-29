"""Data loading and preprocessing utilities."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
try:
    import nltk
    from nltk.corpus import stopwords
    _NLTK_AVAILABLE = True
except Exception:
    _NLTK_AVAILABLE = False

class SpamDataset:
    """Handler for spam email dataset operations."""
    
    def __init__(self, data_path=None):
        """Initialize dataset handler.
        
        Args:
            data_path (str, optional): Path to dataset CSV.
                If None, will download from default URL.
        """
        self.data_path = data_path
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=10000,
            stop_words='english'
        )
    
    def load_data(self):
        """Load and preprocess the spam dataset.
        
        Returns:
            tuple: (X, y) where X is text features and y is labels
        """
        import os

        # Default remote CSV (raw GitHub URL)
        default_url = (
            "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-"
            "Intelligence-for-Cybersecurity/master/Chapter03/datasets/sms_spam_no_header.csv"
        )

        # Ensure data/raw exists
        raw_dir = os.path.join(os.getcwd(), "data", "raw")
        os.makedirs(raw_dir, exist_ok=True)

        # Choose source path or URL
        if self.data_path:
            src = self.data_path
        else:
            src = os.path.join(raw_dir, "spam_dataset.csv")

        # If local copy not present and no explicit data_path, download
        if (not self.data_path) and (not os.path.exists(src)):
            try:
                # Use pandas to read from the remote URL and save locally
                df = pd.read_csv(default_url, header=None, encoding="utf-8")
                df.to_csv(src, index=False, header=False, encoding="utf-8")
            except Exception:
                # Fallback: try reading directly from URL (without saving)
                df = pd.read_csv(default_url, header=None, encoding="utf-8")
        else:
            # Read local file or provided path
            # try with header=None (the dataset has no header)
            try:
                df = pd.read_csv(src, header=None, encoding="utf-8")
            except Exception:
                # Try reading with automatic header detection
                df = pd.read_csv(src, encoding="utf-8")

        # The dataset has two columns: label and message (no header)
        if df.shape[1] >= 2:
            df = df.iloc[:, :2]
            df.columns = ["label", "message"]
        else:
            raise ValueError("Unexpected dataset format: expected at least 2 columns")

        # Normalize labels to binary: spam -> 1, ham/ham-like -> 0
        df["label"] = df["label"].astype(str).str.strip().str.lower()
        df = df[df["message"].notna()]
        y = (df["label"] == "spam").astype(int).to_numpy()
        X = df["message"].astype(str).to_list()

        return X, y
    
    def get_train_test_split(self, test_size=0.2, random_state=42):
        """Get train/test split of vectorized data.
        
        Args:
            test_size (float): Proportion for test set
            random_state (int): Random seed
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        X, y = self.load_data()
        # Preprocess raw text messages before vectorization
        X = self.preprocess_texts(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Fit vectorizer on training data only
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        return X_train_vec, X_test_vec, y_train, y_test

    def _clean_text(self, text: str) -> str:
        """Basic text cleaning pipeline.

        - Lowercase
        - Remove URLs, emails, numbers
        - Remove punctuation
        - Normalize whitespace
        """
        if not isinstance(text, str):
            text = str(text)
        txt = text.lower()
        # remove URLs
        txt = re.sub(r"https?://\S+|www\.\S+", " ", txt)
        # remove email addresses
        txt = re.sub(r"\S+@\S+", " ", txt)
        # remove numbers
        txt = re.sub(r"\d+", " ", txt)
        # remove punctuation
        txt = txt.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
        # collapse whitespace
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt

    def preprocess_texts(self, texts):
        """Preprocess a list of text messages.

        Optionally removes NLTK stopwords if available.
        Returns a list of cleaned (and optionally stopword-filtered) strings.
        """
        cleaned = [self._clean_text(t) for t in texts]

        # Attempt to remove stopwords if NLTK is available; download if necessary
        if _NLTK_AVAILABLE:
            try:
                stops = set(stopwords.words("english"))
            except Exception:
                try:
                    nltk.download("stopwords", quiet=True)
                    stops = set(stopwords.words("english"))
                except Exception:
                    stops = None
            if stops:
                def remove_stops(s):
                    tokens = [w for w in s.split() if w not in stops]
                    return " ".join(tokens)

                cleaned = [remove_stops(s) for s in cleaned]

        return cleaned