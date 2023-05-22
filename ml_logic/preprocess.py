import pandas as pd
import numpy as np
import os
import requests

from PIL import Image

from sklearn.preprocessing import OrdinalEncoder
from tensorflow.keras.utils import to_categorical

from params import TARGET_SIZE, IMAGE_CODES

def download_data(upload_to_cloud=False):

    current_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    raw_images_path = os.path.join(parent_dir, 'pill_pic', 'data', 'raw_images')
    xlsx_path = os.path.join(parent_dir, 'pill_pic', 'data', 'directory_consumer_grade_images.xlsx')

    data = pd.read_excel(xlsx_path)
    data = data[data['NDC11'].isin(IMAGE_CODES)]

    pre_url = 'https://data.lhncbc.nlm.nih.gov/public/Pills/'
    post_url = data['Image']
    full_url = pre_url + post_url

    #key_file_path = os.path.join(os.path.expanduser('~'), ".pill-pic", "pill-pic-a4bcd429b85c.json") # Add the gcloud service acc JSON key to a .pill-pic folder.
    #BUCKET_NAME = "pill_pic_image_set"
    #counter = 1

    #client = storage.Client.from_service_account_json(key_file_path)
    #bucket = client.get_bucket(BUCKET_NAME)

    for index, url in full_url.items():

        file_extension = (os.path.splitext(url)[1]).lower()  # Gets the file extension from the url
        image_path = os.path.join(raw_images_path, f"{index}{file_extension}") # Complete file path including image

        if os.path.exists(image_path):
            #print(f"{index}{file_extension} already exists in {raw_images_path}!")
            continue
        else:
            response = requests.get(url)
            with open(image_path, 'wb') as file:
                file.write(response.content)
            print(f"{index}{file_extension} saved to {raw_images_path}.")

        # Uploads the image to Google Cloud.
        #if upload_to_cloud:
            #blob = bucket.blob(f"{index}{file_extension}")
            #blob.upload_from_filename(image_path)
            #print(f"{index}{file_extension} uploaded to {BUCKET_NAME} on Google Cloud. \n ✅ {counter}/{len(data)}")
            #counter += 1

    print(f"✅ All raw images saved to {raw_images_path}.")


def resize_images():

    current_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    raw_images_path = os.path.join(parent_dir, 'pill_pic', 'data', 'raw_images')
    processed_images_path = os.path.join(parent_dir, 'pill_pic', 'data', 'processed_images')
    xlsx_path = os.path.join(parent_dir, 'pill_pic', 'data', 'directory_consumer_grade_images.xlsx')

    data = pd.read_excel(xlsx_path)
    data = data[data['NDC11'].isin(IMAGE_CODES)]

    resized_images = []
    image_names = []

    for filename in os.listdir(raw_images_path):
        if os.path.exists(os.path.join(processed_images_path, filename)):
            #print(f"{filename} already exists in {processed_images_path}!")
            continue
        elif filename.endswith('.jpg') or filename.endswith('.png'):
            file_path = os.path.join(raw_images_path, filename)

            image = Image.open(file_path)
            rgb_image = image.convert('RGB') # converts image to RGB (jpg -> RGB, png -> RGBA)
            resized_image = rgb_image.resize(TARGET_SIZE) # resizes image to target size
            print(f"This is the resized image: {resized_image}")
            resized_image.save(os.path.join(processed_images_path, filename)) # saves resized image to processed_images_path
            print(f"{filename} resized and saved to {processed_images_path}.")

    for filename in os.listdir(processed_images_path):
        if filename.endswith('.DS_Store'):
            continue  # skip the .DS_Store file on Macs
        file_path = os.path.join(processed_images_path, filename)
        with Image.open(file_path) as image:
            resized_images.append(image)  # records resized images
            image_names.append(filename[:-4])  # records names of resized images

    images_arr = [np.array(image) for image in resized_images]
    images_arr = np.array(images_arr)
    images_arr = images_arr / 255.0 # normalize pixels in all the images


    print(f"✅ {len(images_arr)} images resized to {TARGET_SIZE} and saved to {processed_images_path}.")
    return images_arr, image_names

def get_pill_data(image_names):

    current_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir)) #pill_pic
    xlsx_path = os.path.join(parent_dir, 'pill_pic', 'data', 'directory_consumer_grade_images.xlsx')

    #Create a dataframe using all the data on the pills in IMAGE_CODES
    all_data = pd.read_excel(xlsx_path)
    pill_data = all_data[all_data['NDC11'].isin(IMAGE_CODES)]

    #Create a dataframe using only the images that were finally downloaded and resized
    image_names = [int(name) for name in image_names] #convert image names to int
    index_exists = pill_data.index.isin(image_names) #create a boolean index of the image names that exist in the dataframe
    pill_data = pill_data.loc[index_exists] #create a new dataframe with only the images that have been resized

    print(f"✅ {len(pill_data)} rows of data loaded.")
    return pill_data

def create_and_encode_y(data):

    encoder = OrdinalEncoder()
    data['encoded_NDC11'] = encoder.fit_transform(data[['NDC11']])
    data['encoded_NDC11'] = to_categorical(data['encoded_NDC11'])

    print(f"✅ Target (y) encoded and categorized.")
    return data
