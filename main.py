#basic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#image manipulation
from PIL import Image

#sklearn
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

#tensorflow
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

#created functions
from ml_logic.preprocess import download_images, resize_images, get_pill_data, create_and_encode_y, get_pill_name
from ml_logic.model import initialize_model, compile_model, train_model, evaluate_model
from params import TARGET_SIZE, IMAGE_CODES


def preprocess():
    '''
    Preprocess the images to be ready for use by the model.
    '''
    download_images(upload_to_cloud=False) #downloads raw images to local storage
    images_arr, image_names = resize_images() #resizes images
    pill_data = get_pill_data(image_names) #gets only the data of the images that are used
    pill_data = create_and_encode_y(pill_data) #creates y target and encodes it for further use

    print(f'✅ Resized and preprocessed {images_arr.shape[0]} images in total!')
    print("✅ preprocess() complete.")
    return images_arr, pill_data

def train(images_arr, pill_data, model=None):
    """
    - Get preprocessed images from local storage
    - Train on the preprocessed images
    - Store training results and model weights
    """
    # Create (X_train_processed, y_train, X_val_processed, y_val)
    X_train, X_test, y_train, y_test = train_test_split(images_arr, pill_data['encoded_NDC11'], test_size=0.3)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3)

    # convert y from 0, 1, 2..  to categorical [1, 0, 0], [0, 1, 0], [0, 0, 1]..
    y_train = to_categorical(y_train.values)
    y_val = to_categorical(y_val.values)
    y_test = to_categorical(y_test.values)

    if model is None:
        model = initialize_model()

    model = compile_model(model)

    model, history = train_model(
        model,
        X_train,
        y_train,
        batch_size=16,
        patience=10,
        validation_data=(X_val, y_val)
    )

    val_accuracy = np.min(history.history['val_accuracy'])

    print("✅ train() complete.")
    return model, val_accuracy, X_test, y_test


def evaluate(model, X_test, y_test):
    """
    Evaluate the performance of the latest production model on processed data.
    Return accuracy as a float.
    """

    assert model is not None

    metrics_dict = evaluate_model(model=model, X=X_test, y=y_test)
    accuracy = metrics_dict["accuracy"]

    print("✅ evaluate() complete.")
    return accuracy


def pred(pill_data, model):
    """
    Make a prediction using the latest trained model
    """
    file_path = 'data/uploaded_images/'
    image_files = os.listdir(file_path)

    for file_name in image_files:
        if file_name.endswith('.DS_Store'):
            continue  # skip the .DS_Store file on Macs

        image_path = os.path.join(file_path, file_name)
        print(f"The image that you uploaded is: {file_name}")

        img = Image.open(image_path)
        rgb_image = img.convert('RGB') # converts image to RGB (jpg -> RGB, png -> RGBA)
        resized_image = rgb_image.resize(TARGET_SIZE)

        processed_image = np.array(resized_image) # converts the image to numpy array of shape (224, 224, 3)
        processed_image = np.expand_dims(processed_image, axis=0) # converts the image size to (1, 224, 224, 3)
        processed_image = processed_image / 255.0

        y_pred = model.predict(processed_image) #array of probabilities of the pill being each of the classes
        best_prediction_index = np.argmax(y_pred) #index of the best prediction
        best_prediction_prob = y_pred[0, best_prediction_index] #probability of the pill being best prediction

        pill_name = get_pill_name(best_prediction_index, pill_data)
        print(f"✅ The pill that you uploaded is: {best_prediction_index} {pill_name}, with probability {round(best_prediction_prob * 100, 2)}%")

    print("✅ pred() complete.")
    return y_pred, pill_name

def main():

    images_arr, pill_data = preprocess()

    model, val_accuracy, X_test, y_test = train(images_arr, pill_data)

    test_accuracy = evaluate(model, X_test, y_test)

    y_pred, pill_name = pred(pill_data, model)


if __name__ == '__main__':
    main()
