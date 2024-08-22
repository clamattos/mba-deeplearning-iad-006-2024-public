from fastapi import FastAPI, File
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import pickle
from contextlib import asynccontextmanager
import base64
from PIL import Image
import io

class PredictionResponse(BaseModel):
    prediction: float

class ImageRequest(BaseModel):
    image: str

def load_model():
    global xgb_model_carregado
    with open("models/xgboost.pkl", "rb") as f:
        xgb_model_carregado = pickle.load(f)

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["xgboost"] = load_model()
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/predict")
async def predict(x: float):
    result = ml_models["xgboost"](x)
    return {"result": result}

# Endpoint de Healthcheck
@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}

################# DEFINICAO ENDPOINT PRINCIPAL #################
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ImageRequest):
    img_bytes = base64.b64decode(request.image)
    img = Image.open(io.BytesIO(img_bytes))
    img = img.resize((8, 8))
    img_array = np.array(img)

    img_array = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])

    img_array = img_array.reshape(1, -1)

    prediction = xgb_model_carregado.predict(img_array)

    return {"prediction": int(prediction[0])}
