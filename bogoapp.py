from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as numpy
import pickle
import pandas as pd
# from sklearn.externals import joblib

app = FastAPI()

# Load the trained model from a file
# model = joblib.load("bogo_uplift.pkl")
pickle_in= open("bogo_uplift.pkl","rb")
model = pickle.load(pickle_in)

@app.post("/predict")
def predict(history: int, recency: int, treatment: int):
    # Use the model to make a prediction
    prediction = model.predict([[history, recency, treatment]])
    
    # Return the prediction as a response
    return {"segment": prediction[0]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
    # uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
