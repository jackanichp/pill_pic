import pandas as pd
import numpy as np
import os
import requests

from PIL import Image

from sklearn.preprocessing import OrdinalEncoder

from params import TARGET_SIZE, IMAGE_CODES

def download_images(upload_to_cloud=False):

    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    raw_images_path = os.path.join(parent_dir, 'pill_pic', 'data', 'raw_images')
    xlsx_path = os.path.join(parent_dir, 'pill_pic', 'data', 'directory_consumer_grade_images.xlsx')

    data = pd.read_excel(xlsx_path)
    data = data[data['NDC11'].isin(IMAGE_CODES)]

    pre_url = 'https://data.lhncbc.nlm.nih.gov/public/Pills/'
    post_url = data['Image']
    full_url = pre_url + post_url

    # Info for uploading to Google Cloud
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

    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    raw_images_path = os.path.join(parent_dir, 'pill_pic', 'data', 'raw_images')
    processed_images_path = os.path.join(parent_dir, 'pill_pic', 'data', 'processed_images')
    xlsx_path = os.path.join(parent_dir, 'pill_pic', 'data', 'directory_consumer_grade_images.xlsx')

    data = pd.read_excel(xlsx_path)
    data = data[data['NDC11'].isin(IMAGE_CODES)]

    images_arr = []
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
            image_names.append(filename[:-4])  # records names of resized images
            image = np.array(image)  # converts image to numpy array
            images_arr.append(image)  # adds numpy array image to images_arr

    images_arr = np.array(images_arr)
    images_arr = images_arr / 255.0 # normalize pixels in all the images

    print(f"✅ {len(images_arr)} images resized to {TARGET_SIZE} and saved to {processed_images_path}.")
    return images_arr, image_names

def get_pill_data(image_names):

    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) #pill_pic
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

    #transforms the NDC11 codes into a categorical variable (0, 1, 2, 3...)
    encoder = OrdinalEncoder()
    encoder.fit(data[['NDC11']])
    data['encoded_NDC11'] = encoder.transform(data[['NDC11']])

    print(f"✅ Target (NDC11) ordinally encoded.")
    return data

def get_pill_name(prediction, data):

    name = data.loc[data['encoded_NDC11'] == prediction, 'Name'].iloc[0]

    return name

def create_encoded_csv(filepath):
    # Read the existing data from the Excel file
    data = pd.read_excel(filepath)

    # Filter the data based on the NDC11 codes in IMAGE_CODES
    data = data[data['NDC11'].isin(IMAGE_CODES)]

    # Create a copy of the DataFrame
    new_data = data.copy()

    # Create an instance of the OrdinalEncoder
    encoder = OrdinalEncoder()

    # Fit the encoder on the NDC11 column and transform it
    encoded_NDC11 = encoder.fit_transform(new_data[['NDC11']])

    # Add the encoded_NDC11 as a new column to the DataFrame
    new_data['encoded_NDC11'] = encoded_NDC11

    # Generate the CSV file path
    csv_filepath = filepath.replace('.xlsx', '_encoded.csv')

    # Save the modified DataFrame to a new CSV file
    new_data.to_csv(csv_filepath, index=False)

    print(f"✅ Encoded column appended and saved to {csv_filepath}.")

create_encoded_csv('data/directory_consumer_grade_images.xlsx')
