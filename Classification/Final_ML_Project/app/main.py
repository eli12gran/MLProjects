from fastapi import FastAPI
from model.model import __version__ as model_version
from model.model import predict_pipeline
from pydantic import BaseModel, Field, ValidationError, validator
import pandas as pd
from typing import Optional, Union, List

app = FastAPI()
mean_value_of_C = 39235.33998392166

class Client(BaseModel):
    A: Optional[float]
    B: float = Field(..., description="A float number is required and this field cannot be empty.")
    C: Optional[float] = Field(None, description="A float number is required. Default is the mean of C.")

    @validator("C", pre=True, always=True)
    def set_default_C(cls, value):
        return mean_value_of_C if value is None else value
    D: Optional[float]
    E: Optional[float]
    F: float = Field(default=0, description="A float number is required and this field cannot be empty.")
    G: Optional[float]
    H: Optional[float]
    I: Optional[float]
    J: str = Field(..., min_length=2, max_length=3, description="Country must be between 2 and 3 characters.")
    K: float = Field(default=0, description="This field is optional and accepts a float. Default is 0.")
    L: Optional[float]
    M: Optional[float]
    N: Optional[float]
    O: Optional[float]
    P: float = Field(default=1, description="A float number is required and this field cannot be empty.")
    Q: Union[str, float] = Field(default=0, description="A float or string representation of a number is accepted. Default is 0.")
    R: Optional[Union[str, float]]
    S: float = Field(..., description="A float number is required and this field cannot be empty.")
    Monto: Union[str, float] = Field(..., description="This can be a str or a float value, be careful")

class Prediction(BaseModel):
    Fraud: str

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}


@app.post("/predict", response_model=Prediction)

def predict(client: Client):
    monto_value = client.Monto
    if isinstance(monto_value, str):
        monto_value = float(monto_value.replace(',', ''))
    
    client_data = pd.DataFrame([{
        "A": client.A,
        "B": client.B,
        "C": client.C,
        "D": client.D,
        "E": client.E,
        "F": client.F,
        "G": client.G,
        "H": client.H,
        "I": client.I,
        "J": client.J,
        "K": client.K,
        "L": client.L,
        "M": client.M,
        "N": client.N,
        "O": client.O,
        "P": client.P,
        "Q": client.Q,
        "R": client.R,
        "S": client.S,
        "Monto": monto_value, 
    }])

    fraud = predict_pipeline(client_data)
    
    return {"Fraud": fraud}

