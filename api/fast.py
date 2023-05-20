import pandas as pd
import random
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post('/predict')
def predict(image: UploadFile = File(...)):
    # Compute `wait_prediction` from `day_of_week` and `time`
    drug_prediction = random.choice(["TRI-LEGEST Fe","NORTEL 7/7/7","amrix","TRIVORA TAB","AMARYL 4MG TABLETS"])

    return {'Pill Name': drug_prediction}

@app.get("/")
def root():
    return {'return': 'ok'}
