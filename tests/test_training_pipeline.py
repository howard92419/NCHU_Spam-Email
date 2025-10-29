import os
import numpy as np
from pathlib import Path

from src.models import train
from src.data.dataset import SpamDataset


def test_train_and_save_monkeypatched_dataset(tmp_path, monkeypatch):
    # Prepare a tiny synthetic dataset
    texts = [
        "Win money now",
        "Lowest price for meds",
        "Hi, are we still on for lunch?",
        "Don't miss this offer",
        "Can you review the report?",
        "Reminder: project meeting tomorrow",
    ]
    # labels: spam=1, ham=0
    labels = np.array([1, 1, 0, 1, 0, 0])

    # Monkeypatch SpamDataset.load_data to return our small sample
    def fake_load(self):
        return texts, labels

    monkeypatch.setattr(SpamDataset, "load_data", fake_load)

    # Run training pipeline, saving to tmp path
    model_file = tmp_path / "spam_test_model.joblib"
    metrics, saved_path = train.train_and_save(model_path=str(model_file), test_size=0.5)

    # Check model file exists
    assert os.path.exists(saved_path)

    # Check metrics contain expected keys and sensible ranges
    for key in ("accuracy", "precision", "recall", "f1"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0
