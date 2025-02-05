import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def get_datasets():
    # Load CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Define the labels of the dataset
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']

    # Print dataset shape
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Scale the data
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Transform target variable into one-hot encoding
    y_cat_train = to_categorical(y_train, 10)
    y_cat_test = to_categorical(y_test, 10)

    return (X_train, y_cat_train), (X_test, y_cat_test)  # ✅ Returns dataset

