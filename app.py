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

    return predicted_NDC11, pill_name

def picture_upload(prediction_model):

    # User input fields
    nickname = st.text_input("Nickname")
    age = st.number_input("Age", min_value=0, max_value=120)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    allergy = st.multiselect("Are you allergic to one of the following ingredient?", ["No allergy", "LORAZEPAM","LACTOSE","MAGNESIUM"])
    pregnant = st.selectbox("Do you want to have information about pregnancy compatibility ?", ["Yes", "No"])
    nursing = st.selectbox("Do you want to have information about nursing compatibility ?", ["Yes", "No"])
    kids = st.selectbox("Do you want to have information about pediatric compatibility ?", ["Yes", "No"])

    # Save button
    if st.button("Validate"):
        st.success(f"Welcome to pillpic {nickname}")

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
            predicted_item, predicted_NDC = predict(prediction_model, processed_image, database)

            # Display the uploaded image
            st.image(image, caption="Here's the image you captured ☝️")
            st.success("Image successfully captured and processed!")

            # Display the predicted item
            st.write("Predicted Item:", predicted_item)

            # Display the additionnal informations

            route = data_extension.loc[data_extension.loc['NDC11'] == predicted_NDC,"route"].values[0]
            ingredient = data_extension.loc[data_extension.loc['NDC11'] == predicted_NDC,"spl_product_data_elements"].values[0]
            warning = data_extension.loc[data_extension.loc['NDC11'] == predicted_NDC,"warnings"].values[0]
            indication = data_extension.loc[data_extension.loc['NDC11'] == predicted_NDC,"indications_and_usage"].values[0]
            contra = data_extension.loc[data_extension.loc['NDC11'] == predicted_NDC,"contraindications"].values[0]
            adverse = data_extension.loc[data_extension.loc['NDC11'] == predicted_NDC,"adverse_reactions"].values[0]
            precautions = data_extension.loc[data_extension.loc['NDC11'] == predicted_NDC,"precautions"].values[0]
            dosage = data_extension.loc[data_extension.loc['NDC11'] == predicted_NDC,"dosage_and_administration"].values[0]
            pregnancy = data_extension.loc[data_extension.loc['NDC11'] == predicted_NDC,"pregnancy"].values[0]
            nursing = data_extension.loc[data_extension.loc['NDC11'] == predicted_NDC,"nursing_mothers"].values[0]
            pediatric = data_extension.loc[data_extension.loc['NDC11'] == predicted_NDC,"pediatric_use"].values[0]

            st.markdown(f'Route: {route}')
            st.markdown(f'The pill contains: {ingredient}')

            # Find matching allergies

            list_all = []
            for i in range(0,len(allergy)):
                if allergy[i] in str(ingredient):
                    list_all.append(allergy[i])
            if not list_all:
                pass
            else:
                st.warning(f'Carefull, the pill contains:')
                for j in range(0,len(list_all)):
                    st.warning(allergy[j])

            st.markdown(f'Warnings: {warning}')
            st.markdown(f'Indications & Usages: {indication}')
            st.markdown(f'Containdications: {contra}')
            st.markdown(f'Adverse reactions: {adverse}')
            st.markdown(f'Dosage & administration: {dosage}')
            st.markdown(f'Precautions: {precautions}')
            if pregnant == "Yes":
                st.markdown(f'Pregnancy: {pregnancy}')
            if nursing == "Yes":
                st.markdown(f'Nursing mothers: {nursing}')
            if kids == "Yes":
                st.markdown(f'Pediatric use: {pediatric}')

        except Exception as e:
            st.error("An error occurred during image processing. Please try again.")
            st.error(str(e))

database = pd.read_csv("data/Prediction_df.csv", dtype={"NDC11":str}, low_memory=False).fillna("None")
data_extension = pd.read_csv("data/extended_data.csv", dtype={"NDC11":str}, low_memory=False)
prediction_model = YOLO('best.pt')
detection_model = YOLO('detection.pt', compile=False)
picture_upload(prediction_model)
