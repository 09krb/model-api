from fastapi import FastAPI
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import model_utils

# Lifespan event: load model saat server start
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_utils.load_model()
    yield

app = FastAPI(
    title="Diet Recommendation API",
    description="Prediksi rekomendasi pola makan berdasarkan data harian",
    version="1.0.0",
    lifespan=lifespan
)

# Schema input — Pydantic otomatis validasi tipe & nilai
class DailyInput(BaseModel):
    Age: int = Field(..., ge=0, le=120)
    Height_cm: float = Field(..., ge=50, le=250)
    Weight_kg: float = Field(..., ge=10, le=300)
    Blood_Pressure_Systolic: float = Field(..., ge=60, le=250)
    Blood_Pressure_Diastolic: float = Field(..., ge=40, le=150)
    Cholesterol_Level: float = Field(..., ge=0)
    Blood_Sugar_Level: float = Field(..., ge=0)
    Daily_Steps: int = Field(..., ge=0)
    Exercise_Frequency: int = Field(..., ge=0, le=7)
    Sleep_Hours: float = Field(..., ge=0, le=24)
    Caloric_Intake: float = Field(..., ge=0)
    Protein_Intake: float = Field(..., ge=0)
    Carbohydrate_Intake: float = Field(..., ge=0)
    Fat_Intake: float = Field(..., ge=0)

    # One-hot encoded fields (0 atau 1)
    Gender_Male: int = Field(..., ge=0, le=1)
    Gender_Other: int = Field(..., ge=0, le=1)
    Chronic_Disease_Heart_Disease: int = Field(..., ge=0, le=1, alias="Chronic_Disease_Heart Disease")
    Chronic_Disease_Hypertension: int = Field(..., ge=0, le=1)
    Chronic_Disease_No_Chronic_Disease: int = Field(..., ge=0, le=1, alias="Chronic_Disease_No Chronic Disease")
    Chronic_Disease_Obesity: int = Field(..., ge=0, le=1)
    Genetic_Risk_Factor_Yes: int = Field(..., ge=0, le=1)
    Allergies_Lactose_Intolerance: int = Field(..., ge=0, le=1, alias="Allergies_Lactose Intolerance")
    Allergies_No_Allergies: int = Field(..., ge=0, le=1, alias="Allergies_No Allergies")
    Allergies_Nut_Allergy: int = Field(..., ge=0, le=1, alias="Allergies_Nut Allergy")
    Alcohol_Consumption_Yes: int = Field(..., ge=0, le=1)
    Smoking_Habit_Yes: int = Field(..., ge=0, le=1)
    Dietary_Habits_Regular: int = Field(..., ge=0, le=1)
    Dietary_Habits_Vegan: int = Field(..., ge=0, le=1)
    Dietary_Habits_Vegetarian: int = Field(..., ge=0, le=1)
    Preferred_Cuisine_Indian: int = Field(..., ge=0, le=1)
    Preferred_Cuisine_Mediterranean: int = Field(..., ge=0, le=1)
    Preferred_Cuisine_Western: int = Field(..., ge=0, le=1)
    Food_Aversions_Salty: int = Field(..., ge=0, le=1)
    Food_Aversions_Spicy: int = Field(..., ge=0, le=1)
    Food_Aversions_Sweet: int = Field(..., ge=0, le=1)

    model_config = {"populate_by_name": True}

class PredictionOutput(BaseModel):
    recommendation: str
    confidence: float
    raw_scores: list[float]

@app.get("/")
def root():
    return {"status": "ok", "message": "Diet Recommendation API is running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: DailyInput):
    result = model_utils.predict(input_data.model_dump(by_alias=True))
    return result