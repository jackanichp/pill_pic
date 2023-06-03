import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import torch
import ultralytics
from ultralytics import YOLO

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
        "Pill Pic also allows you to create a user profile \n"
        "so you can keep track of our medication history! \n"
    )
    st.markdown("---")
    st.markdown("A group project by Morgane, Ninaad, Paul and Pierre")

def image_to_square(image):
    background_color = (0,0,0)
    width, height = image.size
    if width == height:
        return image
    elif width > height:
        result = Image.new(image.mode, (width, width), background_color)
        result.paste(image, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(image.mode, (height, height), background_color)
        result.paste(image, ((height - width) // 2, 0))
        return result

def preprocess_image(image):
    # Convert image to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Perform object detection
    results = detection_model.predict(image, conf=0.4)

    # Check if any objects are detected
    if len(results) == 0 or len(results[0].boxes.data) == 0:
        return None

    # Extract bounding box coordinates for the first detected object
    xyxy = results[0].boxes.data[0].tolist()[:4]
    xmin = int(xyxy[0])
    ymin = int(xyxy[1])
    xmax = int(xyxy[2])
    ymax = int(xyxy[3])

    # Crop the image based on the bounding box coordinates
    image = image.crop((xmin, ymin, xmax, ymax))

    #Convert image to square
    square_image = image_to_square(image)

    # Resize image to target size
    TARGET_SIZE = (160, 160)
    square_image = square_image.resize(TARGET_SIZE)

    # Normalize image by dividing by 255
    image_array = np.array(square_image) / 255.0

    return image_array


def get_pill_name(predicted_NDC11, database):
    # Get name of pill
    name = database.loc[database['NDC11'] == predicted_NDC11, 'Name'].iloc[0]
    return name

def predict(prediction_model, processed_image, database):
    # Make the prediction using the model
    prediction = prediction_model.predict(processed_image, imgsz=160, conf=0.5, verbose=False)

    # Get the predicted class index & NDC11
    predicted_NDC11_index = np.argmax(prediction[0].probs.tolist())
    predicted_NDC11 = prediction[0].names[predicted_NDC11_index]

    # Get the pill name from the database
    pill_name = get_pill_name(predicted_NDC11, database)

    return pill_name

def picture_upload(prediction_model):
    st.title("Snap a pic!")

    uploaded_file = st.camera_input("Capture an image")

    if uploaded_file is not None:
        try:
            # Read and preprocess the uploaded image
            image = Image.open(uploaded_file)

            processed_image = preprocess_image(image)

            # Check if no objects are detected
            if processed_image is None:
                st.warning("No pill detected. Please take another photo.")
                return

            # Perform prediction using the model and database
            predicted_item = predict(prediction_model, processed_image, database)

            # Display the uploaded image
            st.image(image, caption="Here's the image you captured ☝️")
            st.success("Image successfully captured and processed!")

            # Display the predicted item
            st.write("Predicted Item:", predicted_item)

        except Exception as e:
            st.error("An error occurred during image processing. Please try again.")
            st.error(str(e))

database = pd.read_csv("data/Merged_data.csv", dtype={"NDC11":str}, low_memory=False)
prediction_model = YOLO('best.pt')
detection_model = YOLO('detection.pt')
picture_upload(prediction_model)
