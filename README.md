# NCHU_Spam-Email

A small research/demo project for spam (SMS/email-like) classification using a baseline SVM model. This repository contains data loading and preprocessing utilities, a TF-IDF + SVM baseline, a training pipeline, evaluation tools, and OpenSpec proposals to track changes.

## Quick links
- Notebook: `notebooks/01_baseline_model.ipynb` (exploration, training, evaluation)
- Training pipeline: `src/models/train.py` (`train_and_save`)
- Dataset & preprocessing: `src/data/dataset.py` (`SpamDataset`)
- Model wrapper: `src/models/classifier.py` (save/load bundle, `load_bundle`)
- Tests: `tests/` (pytest)
- OpenSpec proposals: `openspec/changes/`

---

## Prerequisites
- Python 3.8+ (3.10 recommended)
- Git
- Recommended: create and use the included virtual environment

## Setup (Windows)

1. Create and activate a virtual environment:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies (development + notebook extras):

```cmd
pip install -e .[dev,notebook]
# or, if you prefer requirements.txt
pip install -r requirements.txt
```

## Setup (Unix / macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev,notebook]
```

## Download dataset
The notebook and `SpamDataset` will automatically download and cache the SMS Spam Collection used for the baseline. You can fetch it manually:

```python
import pandas as pd
url = "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/master/Chapter03/datasets/sms_spam_no_header.csv"
df = pd.read_csv(url, header=None)
df.to_csv("data/raw/spam_dataset.csv", index=False, header=False)
```

## Run the baseline notebook
Start Jupyter and open the notebook:

```cmd
jupyter notebook notebooks/01_baseline_model.ipynb
```

Run the notebook cells (Data load → Preprocessing → Train → Evaluate). The notebook includes a demo cell that shows how to load the saved model bundle and run inference.

## Using the CLI tools
The project includes two CLI tools for easy training and prediction:

### Training
Train and save a model bundle using the CLI:

```cmd
# Basic usage (saves to models/trained/spam_classifier)
python scripts/train.py

# Example output:
Loading dataset...
Training model...

Training metrics:
accuracy: 0.9830
precision: 0.9733
recall: 0.9068
f1: 0.9389
confusion_matrix: [[950, 4], [15, 146]]

Model successfully trained and saved to: models/trained/spam_classifier.joblib
```

The model achieves strong performance on the test set with:
- 98.30% accuracy
- 97.33% precision
- 90.68% recall
- 93.89% F1 score

### Prediction
Make predictions on text messages using a trained model:

```cmd
# Example 1: Testing with a spam message
python scripts/predict.py "URGENT! You have won a $1000 gift card. Click here to claim now!"

# Output:
Loading model from models/trained/spam_classifier.joblib...
Prediction: SPAM
Spam probability: 100.00%

# Example 2: Testing with a normal message
python scripts/predict.py "Hey, what time should we meet for lunch tomorrow?"

# Output:
Loading model from models/trained/spam_classifier.joblib...
Prediction: HAM
Spam probability: 0.01%
```

The tools will automatically handle:
- Loading dependencies and data
- Model training and evaluation
- Saving model bundle (classifier + vectorizer + metadata)
- Text preprocessing and prediction
- Probability estimation

Default model location: `models/trained/spam_classifier.joblib`

Custom usage:
```cmd
# Training with custom output location and model name
python scripts/train.py --output-dir custom/path --model-name my_model

# Prediction using custom model
python scripts/predict.py "Your message here" --model-dir custom/path --model-name my_model
```

## Load a saved model bundle (example)

```python
from src.models.classifier import SVMSpamClassifier
clf, vectorizer, metadata = SVMSpamClassifier.load_bundle('models/trained/spam_classifier_svm.joblib')
texts = ["Congratulations! You have won a prize"]
X = vectorizer.transform(texts)
print(clf.predict(X))
```

## Run tests

```cmd
.venv\Scripts\python.exe -m pytest -q
```

## Project structure

```
NCHU_Spam-Email/
├── data/                       # datasets (raw/processed)
├── notebooks/                  # exploration and baseline notebook
├── src/
│   ├── data/                   # data loading and preprocessing
│   ├── models/                 # model wrappers, training pipeline
│   └── evaluation/             # evaluation utilities
├── openspec/                   # OpenSpec proposals & specs
├── tests/                      # pytest tests
├── requirements.txt
├── setup.py
└── README.md
```

## OpenSpec / Change workflow
This repository uses OpenSpec to manage spec-driven changes. Proposals live under `openspec/changes/` and should include `proposal.md`, `tasks.md`, and any spec deltas. See `openspec/AGENTS.md` for authoring rules and validation guidance.

Quick validation:

```cmd
# if you have an `openspec` CLI available
openspec validate <change-id> --strict
```

## Contributing
- Create an OpenSpec proposal for new capabilities or breaking changes
- Implement tasks after proposal approval
- Add tests for new code paths

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Dataset License
The SMS Spam Collection used in this project is publicly available and was created by Tiago A. Almeida and José María Gómez Hidalgo. The dataset is used for research and demonstration purposes.
