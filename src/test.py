import tensorflow as tf
from .dataset import get_datasets
from .utils import load_model

def test_model():
    # Load dataset
    (_, _), (x_test, y_test) = get_datasets()

    # Load trained model
    model = load_model()

    # Evaluate model
    loss, accuracy, precision, recall = model.evaluate(x_test, y_test)
    print(f"\033[32mTest Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}\033[0m")