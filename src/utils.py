import os
import tensorflow as tf

# Function to save model
def save_model(model, path="models/cifar10_model.keras"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"✅ Model saved at {path}")

# Function to load model
def load_model(path="models/cifar10_model.keras"):
    if os.path.exists(path):
        model = tf.keras.models.load_model(path)
        print(f"✅ Model loaded from {path}")
        return model
    else:
        print(f"❌ No saved model found at {path}. Train a model first.")
        return None
