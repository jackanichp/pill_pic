import streamlit as st
import requests
from PIL import Image
from dotenv import load_dotenv
import os
import streamlit as st
import streamlit_authenticator as stauth

st.title("Pill Pℹ️c 💊")

with st.sidebar:
    st.markdown("# About")
    st.markdown(
        "With Pill Pℹ️c, you can snap a photo 📷 of any pill, and \n"
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
    st.markdown("A group project by Morgane, Ninaad and Paul")

def upload_and_store_picture():

    st.title("User Profile")

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
        st.success(f"Welcome to pillpic {nickname}, please upload your pic and we will tailored the pill information to your profile.")

    st.title("Picture Upload")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded file on the server

        st.success("Image successfully uploaded and stored!")

        st.session_state['image'] = uploaded_file

        # API

        url = 'http://localhost:8000'

        col1, col2 = st.columns(2)

        with col1:
            ### Display the image user uploaded
            st.image(Image.open(st.session_state['image']), caption="Here's the image you uploaded")

        with col2:
            with st.spinner("Wait for it..."):

                ### Make request to  API (stream=True to stream response as bytes)
                res = requests.post(url + "/upload_image", files={'img': st.session_state['image']})

                if res.status_code == 200:
                    ### Display the image returned by the API
                    st.markdown(f'Your pill is {res.json()["pill_name"]}')
                    st.markdown(f'Route: {res.json()["route"]}')
                    st.markdown(f'The pill contains: {res.json()["ingredient"]}')

                    list_all = []
                    for i in range(0,len(allergy)):
                        if allergy[i] in str(res.json()["ingredient"]):
                            list_all.append(allergy[i])
                    if not list_all:
                        pass
                    else:
                        st.warning(f'Carefull, the pill contains:')
                        for j in range(0,len(list_all)):
                            st.warning({allergy[j]})

                    st.markdown(f'Warnings: {res.json()["warning"]}')
                    st.markdown(f'Indications & Usages: {res.json()["indication"]}')
                    st.markdown(f'Containdications: {res.json()["contra"]}')
                    st.markdown(f'Adverse reactions: {res.json()["adverse"]}')
                    st.markdown(f'Dosage & administration: {res.json()["dosage"]}')
                    st.markdown(f'Precautions: {res.json()["precautions"]}')
                    if pregnant == "Yes":
                        st.markdown(f'Pregnancy: {res.json()["pregnancy"]}')
                    if nursing == "Yes":
                        st.markdown(f'Nursing mothers: {res.json()["nursing"]}')
                    if kids == "Yes":
                        st.markdown(f'Pediatric use: {res.json()["pediatric"]}')
                else:
                    st.markdown("Something went wrong 😓 Please try again.")
                    print(res.status_code, res.content)

upload_and_store_picture()
