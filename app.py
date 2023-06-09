import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from pillow_heif import register_heif_opener

st.title("Pill Pic 💊")

with st.sidebar:
    st.markdown("# About")
    st.markdown(
        "With Pill Pic, you can snap a photo of any pill, and \n"
        "our object detection and image classification models will \n"
        "recognize the medication and provide you with key information \n"
        "regarding usage and warnings. \n"
        )
    st.markdown(
        "Create a simple user profile to gain keys insights into potential \n"
        "interactions based on allergies, pregnancy, and other relevant info. \n"
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

    image = image.convert('RGB')
    results = detection_model.predict(image, conf=0.4)

    if len(results) == 0 or len(results[0].boxes.data) == 0:
        return None

    # Extract bounding box coordinates for the first detected object
    xyxy = results[0].boxes.data[0].tolist()[:4]
    xmin = int(xyxy[0])
    ymin = int(xyxy[1])
    xmax = int(xyxy[2])
    ymax = int(xyxy[3])

    image = image.crop((xmin, ymin, xmax, ymax))
    image = image_to_square(image)
    image = image.resize((160, 160))
    image = np.array(image)

    #Inception v3 model specific preprocessing steps
    image = image / 255.0
    image -= 0.5
    image *= 2.0

    # Converts the image size to (1, 160, 160, 3) for model input
    image = np.expand_dims(image, axis=0)

    return image

def get_pill_name(best_pred_index, df):

    name = df.iloc[best_pred_index]['Name']
    code = df.iloc[best_pred_index]['NDC11']

    return name, code

def predict(prediction_model, processed_image, df):

    y_pred = prediction_model.predict(processed_image, verbose=[0]) #returns array of probabilities of the pill being each of the classes
    best_pred_index = np.argmax(y_pred) #index of the best prediction

    best_pred_prob = y_pred[0, best_pred_index] #probability of the pill being best prediction
    pill_name, pill_code = get_pill_name(best_pred_index, df)

    print(f" Predicted pill: {best_pred_index} {pill_name}, pill code: {pill_code}, with probability {best_pred_prob}")
    return pill_name, pill_code, best_pred_prob

def picture_upload(prediction_model):

    # User input fields
    nickname = st.text_input("Nickname")
    age = st.number_input("Age", min_value=0, max_value=120)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    allergy = st.multiselect("Are you allergic to any of the following ingredients?", ["No allergy", "LORAZEPAM","LACTOSE","MAGNESIUM"])
    pregnant = st.selectbox("Are you currently pregnant?", ["Yes", "No"])
    nursing = st.selectbox("Are you currently nursing?", ["Yes", "No"])
    kids = st.selectbox("Do you want information for pediatric use (children < 12)?", ["Yes", "No"])
    #kids = "No" if age >= 18 else "Yes"

    # Save button
    if st.button("Validate"):
        st.success(f"Welcome {nickname}.")

    st.title("Snap a pic!")

    uploaded_file = st.camera_input("Capture an image")

    if uploaded_file is not None:
        try:
            # Reads and preprocesses the uploaded image
            register_heif_opener()
            image = Image.open(uploaded_file)
            processed_image = preprocess_image(image)

            # Determine if the pill was detected
            if processed_image is None:
                st.warning("No pill detected. Please take another photo.")
                return None

            # Perform prediction using the classification model, preprocessed image, and the dataframe
            pill_name, pill_code, best_pred_prob = predict(prediction_model, processed_image, df)

            # Display the results
            st.success("Image successfully captured and processed!")
            st.write(f"✅ The pill that you uploaded is: {pill_code} {pill_name}, with probability {round(best_pred_prob * 100, 2)}%\n")

            # Find additional information about the pill using the pill_code and dataframe
            route = df.loc[df['NDC11'] == pill_code, 'route'].iloc[0]
            ingredients = df.loc[df['NDC11'] == pill_code, 'spl_product_data_elements'].iloc[0]
            warnings = df.loc[df['NDC11'] == pill_code, 'warnings'].iloc[0]
            precautions = df.loc[df['NDC11'] == pill_code, 'precautions'].iloc[0]
            indications = df.loc[df['NDC11'] == pill_code, 'indications_and_usage'].iloc[0]
            dosage = df.loc[df['NDC11'] == pill_code, 'dosage_and_administration'].iloc[0]
            contra = df.loc[df['NDC11'] == pill_code, 'contraindications'].iloc[0]
            adverse = df.loc[df['NDC11'] == pill_code, 'adverse_reactions'].iloc[0]
            pharma = df.loc[df['NDC11'] == pill_code, 'clinical_pharmacology'].iloc[0]

            pregnancy = df.loc[df['NDC11'] == pill_code, "pregnancy"].iloc[0]
            nursing = df.loc[df['NDC11'] == pill_code, "nursing_mothers"].iloc[0]
            pediatric = df.loc[df['NDC11'] == pill_code, "pediatric_use"].iloc[0]

            # Find matching allergies
            list_allergies = []
            for i in range(0, len(allergy)):
                if allergy[i] in str(ingredients):
                    list_allergies.append(allergy[i])
            if not list_allergies:
                pass
            else:
                st.warning(f'⚠️ Careful, this pill contains:')
                for j in range(0, len(list_allergies)):
                    st.warning(allergy[j])

            st.markdown(f'Route: {route}')
            st.markdown(f'Ingredients: {ingredients}')

            with st.expander("Warnings"):
                st.write(warnings)
            with st.expander("Precautions"):
                st.write(precautions)
            with st.expander("Indications & Usages"):
                st.write(indications)
            with st.expander("Dosage & Administration"):
                st.write(dosage)
            with st.expander("Contraindications"):
                st.write(contra)
            with st.expander("Adverse reactions"):
                st.write(adverse)
            with st.expander("Clinical Pharmacology"):
                st.write(pharma)

            if pregnant == "Yes":
                st.markdown(f'Pregnancy: {pregnancy}')
            if nursing == "Yes":
                st.markdown(f'Nursing mothers: {nursing}')
            if kids == "Yes":
                st.markdown(f'Pediatric use: {pediatric}')

        except Exception as e:
            st.error("An error occurred during image processing. Please try again.")
            st.error(str(e))


# Dataframe for getting the NDC11, Name, and other information
df = pd.read_csv("data/updated_data.csv", dtype={"NDC11":str}, low_memory=False).fillna("None")

#Detection model
detection_model = YOLO('detection_model.pt')

#Prediction model
prediction_model = load_model("pillpic_model_20230606.h5", compile=False)

# Run the prediction (upload, process, predict, display)
picture_upload(prediction_model)


# For testing in VSCode
#pillname_dict = {0:'172496058', 1:'173024255', 2:'29316013', 3:'39022310', 4:'49702020218',
                 #5:'50111039801', 6:'50111046801', 7:'50419010510', 8:'555032402', 9:'555099702',
                 #10:'57664010488', 11:'591554405', 12:'63459070160', 13:'7365022', 14:'74611413',
                 #15:'93071101', 16:'93213001', 17:'93226801', 18:'93293201', 19:'93725401',
                 #20:'advil', 21:'advil_400', 22:'advil_liqui-gel', 23:'kirkland_acetaminophen', 24:'life_acetaminophen'}

#img_path = "IMG_6267.jpg"
#img = Image.open(img_path)

#detection_model = YOLO('detection_model.pt')

#pp = preprocess_image(img)

#prediction_model = load_model("pillpic_model_20230606.h5", compile=False)
#prediction_model.compile()
#predict(prediction_model, pp, df)
