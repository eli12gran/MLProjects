import __main__
import pickle
from pathlib import Path

__version__ = "0.0.1"

BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f"{BASE_DIR}/meli_pipeline-{__version__}.pkl", "rb") as f:
    model=pickle.load(f)


classes = {1: "papi nos robaron",
           0: "más aprobado que 18 aprobados"}

def predict_pipeline(input):
    pred = model.predict(input)
    return classes[int(pred)]