import argparse
from src.train import train_model
from src.test import test_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Evaluate the model")
    args = parser.parse_args()
    
    if args.train:
        train_model()
    if args.test:
        test_model()