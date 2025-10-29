"""Training pipeline for the spam classifier."""
from pathlib import Path
from typing import Tuple, Dict, Any

from src.data.dataset import SpamDataset
from src.models.classifier import SVMSpamClassifier
from src.evaluation.metrics import evaluate_model


def train_and_save(
    model_path: str | Path = "models/trained/spam_classifier_svm.joblib",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], str]:
    """Train an SVM spam classifier end-to-end and save the model.

    Returns a tuple of (metrics, saved_model_path).
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Load and split data
    ds = SpamDataset()
    X_train, X_test, y_train, y_test = ds.get_train_test_split(
        test_size=test_size, random_state=random_state
    )

    # Initialize and train classifier
    clf = SVMSpamClassifier()
    clf.train(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    try:
        y_proba = clf.predict_proba(X_test)
    except Exception:
        y_proba = None

    metrics = evaluate_model(y_test, y_pred, y_proba)

    # Attach vectorizer and metadata to classifier instance for bundled save
    try:
        clf.vectorizer = ds.vectorizer
    except Exception:
        clf.vectorizer = None

    clf.metadata = {
        "test_size": test_size,
        "random_state": random_state,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }

    # Save bundled model (model + vectorizer + metadata)
    clf.save(str(model_path))

    return metrics, str(model_path)


if __name__ == "__main__":
    m, p = train_and_save()
    print("Saved model to", p)
    print(m)
