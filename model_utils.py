import numpy as np
import tensorflow as tf

model = None

def load_model():
    global model
    model = tf.keras.models.load_model("model/model.keras")
    print("Model loaded.")

def predict(data: dict) -> dict:
    features = np.array([[
        data["Age"],
        data["Height_cm"],
        data["Weight_kg"],
        data["Blood_Pressure_Systolic"],
        data["Blood_Pressure_Diastolic"],
        data["Cholesterol_Level"],
        data["Blood_Sugar_Level"],
        data["Daily_Steps"],
        data["Exercise_Frequency"],
        data["Sleep_Hours"],
        data["Caloric_Intake"],
        data["Protein_Intake"],
        data["Carbohydrate_Intake"],
        data["Fat_Intake"],
        data["Gender_Male"],
        data["Gender_Other"],
        data["Chronic_Disease_Heart Disease"],
        data["Chronic_Disease_Hypertension"],
        data["Chronic_Disease_No Chronic Disease"],
        data["Chronic_Disease_Obesity"],
        data["Genetic_Risk_Factor_Yes"],
        data["Allergies_Lactose Intolerance"],
        data["Allergies_No Allergies"],
        data["Allergies_Nut Allergy"],
        data["Alcohol_Consumption_Yes"],
        data["Smoking_Habit_Yes"],
        data["Dietary_Habits_Regular"],
        data["Dietary_Habits_Vegan"],
        data["Dietary_Habits_Vegetarian"],
        data["Preferred_Cuisine_Indian"],
        data["Preferred_Cuisine_Mediterranean"],
        data["Preferred_Cuisine_Western"],
        data["Food_Aversions_Salty"],
        data["Food_Aversions_Spicy"],
        data["Food_Aversions_Sweet"],
    ]])
    
    output = model.predict(features)
    
    label_map = {0: "High-Protein Diet", 1: "Low-Carb Diet", 2: "Low-Fat Diet"}
    predicted_class = int(np.argmax(output[0]))
    confidence = float(np.max(output[0]))
    
    return {
        "recommendation": label_map[predicted_class],
        "confidence": round(confidence, 4),
        "raw_scores": output[0].tolist()
    }