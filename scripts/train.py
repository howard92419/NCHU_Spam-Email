#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

# Add the parent directory to the Python path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import SpamDataset
from src.models.train import train_and_save

def main():
    parser = argparse.ArgumentParser(description='Train the spam classifier model')
    parser.add_argument('--output-dir', 
                       type=str,
                       default='models/trained',
                       help='Directory to save the trained model (default: models/trained)')
    parser.add_argument('--model-name',
                       type=str,
                       default='spam_classifier',
                       help='Name of the model bundle (default: spam_classifier)')
    
    args = parser.parse_args()
    
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading dataset...")
    dataset = SpamDataset()
    
    print("Training model...")
    model_path = Path(args.output_dir) / f"{args.model_name}.joblib"
    metrics, saved_path = train_and_save(model_path)
    print("\nTraining metrics:")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    print(f"\nModel successfully trained and saved to: {saved_path}")

if __name__ == '__main__':
    main()