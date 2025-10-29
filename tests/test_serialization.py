import os
import numpy as np

from src.models import train
from src.models.classifier import SVMSpamClassifier
from src.data.dataset import SpamDataset


def test_save_and_load_bundle_and_predict(tmp_path, monkeypatch):
    # Tiny toy dataset
    texts = [
        "Free money now",
        "Win a prize",
        "Let's have lunch tomorrow",
        "Can you review this document",
    ]
    labels = np.array([1, 1, 0, 0])

    def fake_load(self):
        return texts, labels

    monkeypatch.setattr(SpamDataset, "load_data", fake_load)

    model_file = tmp_path / "bundle.joblib"

    metrics, saved_path = train.train_and_save(model_path=str(model_file), test_size=0.5)

    # Load using convenience loader
    clf, vec, meta = SVMSpamClassifier.load_bundle(saved_path)

    assert hasattr(clf, "model")
    assert vec is not None, "Vectorizer should be saved and restored"
    assert isinstance(meta, dict)

    # Make a small prediction using restored vectorizer and classifier
    sample = ["Congratulations, you won a free ticket"]
    Xs = vec.transform(sample)
    preds = clf.predict(Xs)

    assert len(preds) == 1
    assert os.path.exists(saved_path)
