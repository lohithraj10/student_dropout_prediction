from pathlib import Path
import pickle

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load the saved model pipeline from the same folder as main.py
MODEL_PATH = Path(__file__).with_name("student_dropout_model.pkl")

with MODEL_PATH.open("rb") as file:
    model = pickle.load(file)

# Exact input columns and their order, taken from the saved pipeline
FEATURES = list(model.feature_names_in_)

app = FastAPI(title="Student Dropout Prediction API")


class StudentData(BaseModel):
    marital_status: float
    application_mode: float
    application_order: float
    course: float
    daytime_evening_attendance: float
    previous_qualification: float
    previous_qualification_grade: float
    nacionality: float
    mothers_qualification: float
    fathers_qualification: float
    mothers_occupation: float
    fathers_occupation: float
    admission_grade: float
    displaced: float
    educational_special_needs: float
    debtor: float
    tuition_fees_up_to_date: float
    gender: float
    scholarship_holder: float
    age_at_enrollment: float
    international: float
    curricular_units_1st_sem_credited: float
    curricular_units_1st_sem_enrolled: float
    curricular_units_1st_sem_evaluations: float
    curricular_units_1st_sem_approved: float
    curricular_units_1st_sem_grade: float
    curricular_units_1st_sem_without_evaluations: float
    unemployment_rate: float
    inflation_rate: float
    gdp: float


# Maps the API's JSON field names to the exact column names the model was trained on
ALIASES = {
    "marital_status": "Marital status",
    "application_mode": "Application mode",
    "application_order": "Application order",
    "course": "Course",
    "daytime_evening_attendance": "Daytime/evening attendance",
    "previous_qualification": "Previous qualification",
    "previous_qualification_grade": "Previous qualification (grade)",
    "nacionality": "Nacionality",
    "mothers_qualification": "Mother's qualification",
    "fathers_qualification": "Father's qualification",
    "mothers_occupation": "Mother's occupation",
    "fathers_occupation": "Father's occupation",
    "admission_grade": "Admission grade",
    "displaced": "Displaced",
    "educational_special_needs": "Educational special needs",
    "debtor": "Debtor",
    "tuition_fees_up_to_date": "Tuition fees up to date",
    "gender": "Gender",
    "scholarship_holder": "Scholarship holder",
    "age_at_enrollment": "Age at enrollment",
    "international": "International",
    "curricular_units_1st_sem_credited": "Curricular units 1st sem (credited)",
    "curricular_units_1st_sem_enrolled": "Curricular units 1st sem (enrolled)",
    "curricular_units_1st_sem_evaluations": "Curricular units 1st sem (evaluations)",
    "curricular_units_1st_sem_approved": "Curricular units 1st sem (approved)",
    "curricular_units_1st_sem_grade": "Curricular units 1st sem (grade)",
    "curricular_units_1st_sem_without_evaluations":
        "Curricular units 1st sem (without evaluations)",
    "unemployment_rate": "Unemployment rate",
    "inflation_rate": "Inflation rate",
    "gdp": "GDP",
}


@app.get("/")
def home():
    return {"message": "Student Dropout Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(student: StudentData):
    # Convert API-friendly names to the exact dataset column names
    values = student.model_dump()
    row = {ALIASES[key]: value for key, value in values.items()}

    # Arrange columns in the exact order used during training
    input_df = pd.DataFrame([row], columns=FEATURES)

    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0, 1])

    return {
        "prediction": prediction,
        "status": "Dropout" if prediction == 1 else "Not Dropout",
        "dropout_probability": round(probability, 4),
    }
