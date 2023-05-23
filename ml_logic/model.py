import numpy as np
import time

from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping

from params import IMAGE_CODES

def initialize_model():

    # model for testing
    model = Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3), padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(5, activation='softmax')) # 5 classes can be replaced with len(IMAGE_CODES)

    #  model with ~80% accuracy
    '''model = Sequential()

    model.add(layers.Conv2D(64, (3,3), input_shape=(224, 224, 3), padding='same', activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(layers.Conv2D(64, (3,3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(layers.Conv2D(32, (2,2), activation="relu"))

    model.add(layers.Flatten())

    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(len(IMAGE_CODES), activation='softmax'))'''

    print("✅ Model initialized sucessfully.")
    return model

def compile_model(model):
    '''
    Compiles the Neural Network
    '''
    model.compile(
      loss='categorical_crossentropy',
      optimizer='adam',
      metrics=['accuracy']
    )

    print("✅ Model compiled sucessfully.")
    return model

def train_model(
        model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=16,
        patience=10,
        validation_data=None, # overrides validation_split
        validation_split=0.3
    ):
    """
    Fit the model and return a tuple (fitted_model, history)
    """

    es = EarlyStopping(
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=1000,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1
    )

    print(f"✅ Model trained on {len(X)} rows with a minimum validation accuracy of: {round(np.min(history.history['val_accuracy']), 5)}")

    return model, history


def evaluate_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=64
    ):
    """
    Evaluate trained model performance on the dataset
    """

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=0,
        #callbacks=None,
        return_dict=True
    )

    loss = metrics["loss"]
    mae = metrics["accuracy"]

    print(f"✅ Model evaluated sucessfully, the test accuracy is: {round(mae, 5)}.")

    return metrics
