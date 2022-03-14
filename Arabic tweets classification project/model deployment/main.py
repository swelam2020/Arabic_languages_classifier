from fastapi import FastAPI
import uvicorn
import numpy as np
import pandas as pd
import pickle
from sentence import sent

app = FastAPI()
pickle_in = open('model.pkl', 'rb')
classifier = pickle.load(pickle_in)


@app.get("/")
async def root():
    return {"message": "Hello from the language classifier API"}


@app.post("/predict")
def predict_lang(data: list[str]):
    #data = data.dict()
    # s=sent
    pred = classifier.predict(data)
    return {'prediction': pred}
