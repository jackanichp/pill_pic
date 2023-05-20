import streamlit as st
import requests
from PIL import Image
from dotenv import load_dotenv
import os

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
            st.image(Image.open(st.session_state['image']), caption="Here's the image you uploaded ☝️")

        with col2:
            with st.spinner("Wait for it..."):

                ### Make request to  API (stream=True to stream response as bytes)
                res = requests.post(url + "/upload_image", files={'img': st.session_state['image']})

                if res.status_code == 200:
                    ### Display the image returned by the API
                    st.markdown(f'Your pill is {res.json()["pill_name"]}')
                else:
                    st.markdown("Something went wrong 😓 Please try again.")
                    print(res.status_code, res.content)

# Usage
upload_and_store_picture()
