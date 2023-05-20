import pandas as pd
import random
import numpy as np
import cv2
import io

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
from io import BytesIO
from PIL import Image

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post('/upload_image')
async def predict(img: UploadFile = File(...)):

    # Return random drug name
    drug_prediction = random.choice(["TRI-LEGEST Fe","NORTEL 7/7/7","amrix","TRIVORA TAB","AMARYL 4MG TABLETS"])

    # Encoding and responding with the image
    contents = await img.read()
    '''image_io = BytesIO(contents)
    image_pil = Image.open(image_io)
    cv2_img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)'''

    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray

    return {"pill_name": drug_prediction}

@app.get("/")
def root():
    return {'return': 'ok'}
