import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

st.title("Pill Pic 💊")

with st.sidebar:
    st.markdown("# About")
    st.markdown(
        "With Pill Pic, you can snap a photo 📷 of any pill, and \n"
        "our image classification model trained on over 130K 🖼️ \n"
        "images will recognize the pill type, dosage and manufacturer! \n"
        "In addition, we'll supply you with detailed information about \n"
        "your medication, such as possible interactions, allergies."
        )
    st.markdown(
        "Pill Pℹ️c also allows you to create a user profile \n"
        "so you can keep track of our medication history! \n"
    )
    st.markdown("---")
    st.markdown("A group project by Morgane, Ninaad, Paul and Pierre")

TARGET_SIZE = (224, 224)

def preprocess_image(image):
    # Convert image to PIL image if it's a NumPy array
    if isinstance(image, np.ndarray):
        # Ensure image has the correct shape and data type
        if len(image.shape) == 3 and image.shape[2] == 3 and image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        else:
            raise ValueError("Invalid image shape or data type.")

    # Convert image to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize image to target size
    image = image.resize(TARGET_SIZE)

    # Normalize image by dividing by 255
    image_array = np.array(image) / 255.0

    return image_array

def get_pill_name(prediction, data):
    # Get name of pill
    name = data.loc[data['encoded_NDC11'] == prediction, 'Name'].iloc[0]
    return name

def predict(model, processed_image, database):
    # Ensure the processed image has the correct shape
    if len(processed_image.shape) == 3:
        processed_image = np.expand_dims(processed_image, axis=0)

    # Make the prediction using the model
    prediction = model.predict(processed_image)

    # Get the predicted class index
    predicted_class = np.argmax(prediction)

    # Get the pill name from the database
    pill_name = get_pill_name(predicted_class, database)

    return pill_name

def picture_upload(model, database):
    st.title("Picture Upload")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read and preprocess the uploaded image
        img = Image.open(uploaded_file)
        processed_image = preprocess_image(img)

        # Perform prediction using the model and database
        predicted_item = predict(model, processed_image, database)

        # Display the uploaded image
        st.image(img, caption="Here's the image you uploaded ☝️")
        st.success("Image successfully uploaded and processed!")

        # Display the predicted item
        st.write("Predicted Item:", predicted_item)

# Usage
model = load_model("model.h5")
database = pd.read_csv("data/directory_consumer_grade_images_encoded.csv")
picture_upload(model, database)
