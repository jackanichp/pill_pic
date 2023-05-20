import streamlit as st

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
        with open(uploaded_file.name, "wb") as file:
            file.write(uploaded_file.read())

        st.success("Image successfully uploaded and stored!")

# Usage
upload_and_store_picture()
