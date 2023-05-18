import streamlit as st

def set_background_color(color):
    hex_color = '#%02x%02x%02x' % color
    css = f"""
        <style>
        body {{
            background-color: {hex_color};
        }}
        </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def upload_and_store_picture():
    st.title("Picture Upload")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Save the uploaded file on the server
        with open(uploaded_file.name, "wb") as file:
            file.write(uploaded_file.read())
        st.success("Image successfully uploaded and stored!")

# Set the background color to light blue
set_background_color((173, 216, 230))

# Usage
upload_and_store_picture()
