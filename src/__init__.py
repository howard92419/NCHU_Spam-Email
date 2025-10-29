"""Spam email classification package."""

from src.data.dataset import SpamDataset
from src.models.classifier import SVMSpamClassifier
from src.evaluation.metrics import evaluate_model

__version__ = "0.1.0"

__all__ = ["SpamDataset", "SVMSpamClassifier", "evaluate_model"]