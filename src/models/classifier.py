"""Spam classification models."""

from sklearn.svm import SVC
import joblib

class SVMSpamClassifier:
    """Support Vector Machine classifier for spam detection."""
    
    def __init__(self, **kwargs):
        """Initialize SVM classifier.
        
        Args:
            **kwargs: Arguments passed to sklearn.svm.SVC
        """
        self.model = SVC(
            kernel='linear',
            probability=True,
            **kwargs
        )
    
    def train(self, X, y):
        """Train the classifier.
        
        Args:
            X: Feature matrix
            y: Target labels
        """
        self.model.fit(X, y)
    
    def predict(self, X):
        """Predict spam probability for new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            array: Predicted probabilities
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            array: Class probabilities
        """
        return self.model.predict_proba(X)
    
    def save(self, path):
        """Save model to disk.
        
        Args:
            path (str): Path to save model file
        """
        # Ensure parent directory exists
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Save a bundle containing model and optional vectorizer/metadata if present
        bundle = {"model": self.model}
        # optional attributes that may be attached to the instance
        if hasattr(self, "vectorizer") and self.vectorizer is not None:
            bundle["vectorizer"] = self.vectorizer
        if hasattr(self, "metadata") and self.metadata is not None:
            bundle["metadata"] = self.metadata

        joblib.dump(bundle, str(p))
    
    @classmethod
    def load(cls, path):
        """Load model from disk.
        
        Args:
            path (str): Path to model file
            
        Returns:
            SVMSpamClassifier: Loaded model
        """
        instance = cls()
        loaded = joblib.load(path)

        # Support both legacy direct-model files and new bundle format
        if isinstance(loaded, dict):
            instance.model = loaded.get("model")
            instance.vectorizer = loaded.get("vectorizer")
            instance.metadata = loaded.get("metadata")
        else:
            instance.model = loaded
            instance.vectorizer = None
            instance.metadata = None

        return instance

    @classmethod
    def load_bundle(cls, path):
        """Convenience loader that returns (instance, vectorizer, metadata).

        This wraps `load()` and returns the classifier instance plus extracted
        vectorizer and metadata for ease of use in inference pipelines.
        """
        inst = cls.load(path)
        vec = getattr(inst, "vectorizer", None)
        meta = getattr(inst, "metadata", None)
        return inst, vec, meta