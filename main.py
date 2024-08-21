from fastapi import FastAPI, File
from pydantic import BaseModel
import xgboost as xgb
import numpy as np 
import pickle
import base64
import io
from PIL import Image

app = FastAPI()

