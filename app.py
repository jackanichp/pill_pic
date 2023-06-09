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
        "With Pill Pic, you can snap a photo 📷 of any pill, and \n"
        "our object detection and image classification models will 🖼️ \n"
        "recognize the medication and provide you with key information \n"
        "regarding usage and warnings. \n"
        )
    st.markdown(
        "Create a simple user profile to gain keys insights into potential \n"
        "interactions based on allergies, pregnancy and other relevant info. \n"
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
    # Crop the image based on the bounding box coordinates
    image = image.crop((xmin, ymin, xmax, ymax))

    image = image_to_square(image)

    image = image.resize((160, 160)) # resized the image to (160, 160)

    image = np.array(image)

    image = image / 255.0
    image -= 0.5 #inception v3 specific
    image *= 2.0 #inception v3 specific

    image = np.expand_dims(image, axis=0) # converts the image size to (1, 160, 160, 3)

    return image

def get_pill_name(predicted_pill, database):
    # Get name of pill
    name = database.loc[database['NDC11'] == predicted_pill, 'Name'].iloc[0]
    return name

def predict(prediction_model, processed_image):
    # Make the prediction using the model
    y_pred = prediction_model.predict(processed_image, verbose=[0]) #array of probabilities of the pill being each of the classes
    best_prediction_index = np.argmax(y_pred) #index of the best prediction
    best_prediction_prob = y_pred[0, best_prediction_index] #probability of the pill being best prediction
    predicted_pill = get_pill_name(predicted_pill, database)

    # predicted_NDC11 = pillname_dict[best_prediction_index]

    print(f"✅ The pill that you uploaded is: {best_prediction_index} {predicted_pill}, with probability {round(best_prediction_prob * 100, 2)}%\n")

    return predicted_pill

    # Get the predicted class index & NDC11
    # predicted_NDC11_index = np.argmax(prediction[0].probs.tolist())
    # predicted_NDC11 = prediction[0].names[predicted_NDC11_index]

    # Get the pill name from the database
    # pill_name = get_pill_name(predicted_NDC11, database)

    # return predicted_NDC11, pill_name

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
            # if uploaded_file.endswith('.HEIC'):
            register_heif_opener()
            image = Image.open(uploaded_file)

            # else:
            #     image = np.array(Image.open(uploaded_file))

            processed_image = preprocess_image(image)

            # Check if no objects are detected
            if processed_image is None:
                st.warning("No pill detected. Please take another photo.")
                return

            # Perform prediction using the model and database
            # predicted_item, predicted_NDC = predict(prediction_model, processed_image)
            predicted_NDC = predict(prediction_model, processed_image)

            # Display the uploaded image
            st.image(image, caption="Here's the image you captured ☝️")
            st.write("This is the name of your image:", image)
            st.success("Image successfully captured and processed!")

            # Display the predicted item
            st.write("Predicted Item:", predicted_NDC)

            # Display the additionnal informations

            # route = data_extension.loc[data_extension.loc['NDC11'] == predicted_NDC,"route"].values[0]
            # ingredient = data_extension.loc[data_extension.loc['NDC11'] == predicted_NDC,"spl_product_data_elements"].values[0]
            # warning = data_extension.loc[data_extension.loc['NDC11'] == predicted_NDC,"warnings"].values[0]
            # indication = data_extension.loc[data_extension.loc['NDC11'] == predicted_NDC,"indications_and_usage"].values[0]
            # contra = data_extension.loc[data_extension.loc['NDC11'] == predicted_NDC,"contraindications"].values[0]
            # adverse = data_extension.loc[data_extension.loc['NDC11'] == predicted_NDC,"adverse_reactions"].values[0]
            # precautions = data_extension.loc[data_extension.loc['NDC11'] == predicted_NDC,"precautions"].values[0]
            # dosage = data_extension.loc[data_extension.loc['NDC11'] == predicted_NDC,"dosage_and_administration"].values[0]
            # pregnancy = data_extension.loc[data_extension.loc['NDC11'] == predicted_NDC,"pregnancy"].values[0]
            # nursing = data_extension.loc[data_extension.loc['NDC11'] == predicted_NDC,"nursing_mothers"].values[0]
            # pediatric = data_extension.loc[data_extension.loc['NDC11'] == predicted_NDC,"pediatric_use"].values[0]

            # st.markdown(f'Route: {route}')
            # st.markdown(f'The pill contains: {ingredient}')

            # Find matching allergies

            # list_all = []
            # for i in range(0,len(allergy)):
            #     if allergy[i] in str(ingredient):
            #         list_all.append(allergy[i])
            # if not list_all:
            #     pass
            # else:
            #     st.warning(f'Carefull, the pill contains:')
            #     for j in range(0,len(list_all)):
            #         st.warning(allergy[j])

            # st.markdown(f'Warnings: {warning}')
            # st.markdown(f'Indications & Usages: {indication}')
            # st.markdown(f'Containdications: {contra}')
            # st.markdown(f'Adverse reactions: {adverse}')
            # st.markdown(f'Dosage & administration: {dosage}')
            # st.markdown(f'Precautions: {precautions}')
            # if pregnant == "Yes":
            #     st.markdown(f'Pregnancy: {pregnancy}')
            # if nursing == "Yes":
            #     st.markdown(f'Nursing mothers: {nursing}')
            # if kids == "Yes":
            #     st.markdown(f'Pediatric use: {pediatric}')

        except Exception as e:
            st.error("An error occurred during image processing. Please try again.")
            st.error(str(e))

pillname_dict = {0:'172496058', 1:'173024255', 2:'29316013', 3:'39022310', 4:'49702020218',
                 5:'50111039801', 6:'50111046801', 7:'50419010510', 8:'555032402', 9:'555099702',
                 10:'57664010488', 11:'591554405', 12:'63459070160', 13:'7365022', 14:'74611413',
                 15:'93071101', 16:'93213001', 17:'93226801', 18:'93293201', 19:'93725401',
                 20:'advil', 21:'advil_400', 22:'advil_liqui-gel', 23:'kirkland_acetaminophen', 24:'life_acetaminophen'}

database = pd.read_csv("data/extended_data.csv", dtype={"NDC11":str}, low_memory=False).fillna("None")
# prediction_model = YOLO('best.pt')
prediction_model = load_model("pillpic_model_20230606.h5", compile=False)
prediction_model.compile()
detection_model = YOLO('detection.pt')
picture_upload(prediction_model)

# testing vscode
img_path="118297.jpg"
img=Image.open(img_path)
detection_model = YOLO('detection.pt')
# testing vscode
pp = preprocess_image(img)
# testing vscode
prediction_model = load_model("pillpic_model_20230606.h5", compile=False)
predict(prediction_model, pp)
