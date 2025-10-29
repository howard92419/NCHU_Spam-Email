#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Add the parent directory to the Python path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from src.models.classifier import SVMSpamClassifier

def main():
    parser = argparse.ArgumentParser(description='Make predictions using the trained spam classifier')
    parser.add_argument('text',
                       type=str,
                       help='Text message to classify')
    parser.add_argument('--model-dir',
                       type=str,
                       default='models/trained',
                       help='Directory containing the trained model (default: models/trained)')
    parser.add_argument('--model-name',
                       type=str,
                       default='spam_classifier',
                       help='Name of the model bundle to use (default: spam_classifier)')
    
    args = parser.parse_args()
    
    # Load the model
    model_path = Path(args.model_dir) / f"{args.model_name}.joblib"
    print(f"Loading model from {model_path}...")
    classifier, vectorizer, _ = SVMSpamClassifier.load_bundle(str(model_path))
    
    # Vectorize input and make prediction
    X = vectorizer.transform([args.text])
    prediction = classifier.predict(X)[0]
    probability = classifier.predict_proba(X)[0][1]
    
    # Print results
    result = "SPAM" if prediction == 1 else "HAM"
    print(f"\nPrediction: {result}")
    print(f"Spam probability: {probability:.2%}")

if __name__ == '__main__':
    main()