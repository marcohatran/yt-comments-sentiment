# EDA Pkgs
import pandas as pd
import numpy as np
from datetime import datetime

# Utils
import joblib
pipe_lr = joblib.load(
    open("services/models/emotion_detection_in_text_pipe_lr.pkl", "rb"))


# Fxn
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results