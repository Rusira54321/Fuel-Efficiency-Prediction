from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

model = joblib.load("mpg_model.pkl")
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://fuel-efficiency-prediction-frontend-jyinypalu.vercel.app"],  # Replace * with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Define input data structure
class CarFeatures(BaseModel):
    horsepower: float
    displacement: float
    weight_kg: float
    acceleration: float
    model_year: float
    Europe: int
    Japan: int
    USA: int
    brand_ford: int
    brand_chevrolet: int
    brand_toyota: int
    brand_audi:int
    brand_buick:int
    brand_chrysler:int
    brand_datsun:int
    brand_dodge:int
    brand_fiat:int
    brand_honda:int
    brand_mazda:int
    brand_mercury:int
    brand_oldsmobile:int
    brand_opel:int
    brand_other:int
    brand_peugeot:int
    brand_plymouth:int
    brand_pontiac:int
    brand_renault:int
    brand_saab:int
    brand_subaru:int
    brand_volkswagen:int
    brand_volvo:int
    cyl_3:int
    cyl_4: int
    cyl_5: int
    cyl_6: int
    cyl_8: int


@app.post("/predict")
def predict_mpg(data: CarFeatures):
    input_data = pd.DataFrame([data.dict()])
    prediction = model.predict(input_data)[0]
    return {"mpg_prediction": round(prediction, 2)}