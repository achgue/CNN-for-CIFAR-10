import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .model import build_model
from .dataset import get_datasets
from .utils import save_model

def train_model():
    # Load dataset
    (x_train, y_train), (x_test, y_test) = get_datasets()

    # Data augmentation
    data_generator = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    train_generator = data_generator.flow(x_train, y_train, batch_size=32)
    steps_per_epoch = x_train.shape[0] // 32  # batch_size = 32

    # Build model
    model = build_model()
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    # Early stopping callback
    early_stop = EarlyStopping(monitor='val_loss', patience=4)

    # Train the model
    history = model.fit(
        train_generator,
        epochs=50,
        steps_per_epoch=steps_per_epoch,
        validation_data=(x_test, y_test),
        callbacks=[early_stop]
    )

    # Save the trained model
    save_model(model)