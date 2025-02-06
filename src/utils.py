import os
import tensorflow as tf

# Function to save model
def save_model(model, path="models/cifar10_model.keras"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"\033[32m✅ Model saved at {path}\033[0m")

# Function to load model
def load_model(path="models/cifar10_model.keras"):
    if os.path.exists(path):
        model = tf.keras.models.load_model(path)
        print(f"\033[32m✅ Model loaded from {path}\033[0m")
        return model
    else:
        print(f"\033[31m❌ No saved model found at {path}. Train a model first.\033[0m")
        return None
