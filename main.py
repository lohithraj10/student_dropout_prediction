from pathlib import Path
import pickle
from typing import Dict, List, Optional
 
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
 
# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).with_name("student_dropout_model.pkl")
 
with MODEL_PATH.open("rb") as file:
    model = pickle.load(file)
 
FEATURES = list(model.feature_names_in_)
CLASSES = list(getattr(model, "classes_", [0, 1]))
 
 
def class_label(c, n_classes: int) -> str:
    """Turn a model class (string or int) into a readable label."""
    if isinstance(c, str):
        s = c.lower()
        if "drop" in s:
            return "Dropout"
        if "enrol" in s:
            return "Enrolled"
        if "grad" in s:
            return "Graduate"
        return c
    if n_classes == 2:
        return {0: "Not Dropout", 1: "Dropout"}.get(int(c), str(c))
    # 3-class models (UCI dataset, alphabetical label encoding)
    return {0: "Dropout", 1: "Enrolled", 2: "Graduate"}.get(int(c), str(c))
 
 
CLASS_LABELS = [class_label(c, len(CLASSES)) for c in CLASSES]
IS_MULTICLASS = len(CLASSES) > 2
 
app = FastAPI(
    title="Student Dropout Prediction API",
    description=(
        "Predicts a student's outcome, explains the risk factors and suggests "
        "support actions. Includes a what-if simulator that shows how much "
        "each action could reduce the dropout risk."
    ),
    version="2.0.0",
)
 
# ---------------------------------------------------------------------------
# Input schema (with types, ranges, descriptions and examples)
# ---------------------------------------------------------------------------
SAMPLE_INPUT = {
    "marital_status": 1,
    "application_mode": 17,
    "application_order": 5,
    "course": 171,
    "daytime_evening_attendance": 1,
    "previous_qualification": 1,
    "previous_qualification_grade": 122.0,
    "nacionality": 1,
    "mothers_qualification": 19,
    "fathers_qualification": 12,
    "mothers_occupation": 5,
    "fathers_occupation": 9,
    "admission_grade": 127.3,
    "displaced": 1,
    "educational_special_needs": 0,
    "debtor": 0,
    "tuition_fees_up_to_date": 1,
    "gender": 1,
    "scholarship_holder": 0,
    "age_at_enrollment": 20,
    "international": 0,
    "curricular_units_1st_sem_credited": 0,
    "curricular_units_1st_sem_enrolled": 6,
    "curricular_units_1st_sem_evaluations": 6,
    "curricular_units_1st_sem_approved": 6,
    "curricular_units_1st_sem_grade": 14.0,
    "curricular_units_1st_sem_without_evaluations": 0,
    "unemployment_rate": 13.9,
    "inflation_rate": -0.3,
    "gdp": 0.79,
}
 
 
class StudentData(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": SAMPLE_INPUT})
 
    marital_status: int = Field(..., ge=1, le=6, examples=[1],
        description="1=single, 2=married, 3=widower, 4=divorced, 5=facto union, 6=legally separated")
    application_mode: int = Field(..., ge=1, le=57, examples=[17],
        description="Application mode code (1-57), e.g. 1=1st phase general, 17=2nd phase general, 39=over 23 years old")
    application_order: int = Field(..., ge=0, le=9, examples=[5],
        description="Choice order of the course: 0 = first choice ... 9 = last choice")
    course: int = Field(..., ge=33, le=9991, examples=[171],
        description="Course code, e.g. 33, 171, 8014, 9003, 9070, 9085, 9119, 9130, 9147, 9238, 9254, 9500, 9556, 9670, 9773, 9853, 9991")
    daytime_evening_attendance: int = Field(..., ge=0, le=1, examples=[1],
        description="1 = daytime, 0 = evening")
    previous_qualification: int = Field(..., ge=1, le=43, examples=[1],
        description="Previous qualification code (1-43), 1 = secondary education")
    previous_qualification_grade: float = Field(..., ge=0, le=200, examples=[122.0],
        description="Grade of previous qualification (0-200)")
    nacionality: int = Field(..., ge=1, le=109, examples=[1],
        description="Nationality code (1-109), 1 = Portuguese")
    mothers_qualification: int = Field(..., ge=1, le=44, examples=[19],
        description="Mother's qualification code (1-44)")
    fathers_qualification: int = Field(..., ge=1, le=44, examples=[12],
        description="Father's qualification code (1-44)")
    mothers_occupation: int = Field(..., ge=0, le=195, examples=[5],
        description="Mother's occupation code (0-195)")
    fathers_occupation: int = Field(..., ge=0, le=195, examples=[9],
        description="Father's occupation code (0-195)")
    admission_grade: float = Field(..., ge=0, le=200, examples=[127.3],
        description="Admission grade (0-200)")
    displaced: int = Field(..., ge=0, le=1, examples=[1],
        description="1 = living away from home (displaced), 0 = no")
    educational_special_needs: int = Field(..., ge=0, le=1, examples=[0],
        description="1 = has special needs, 0 = no")
    debtor: int = Field(..., ge=0, le=1, examples=[0],
        description="1 = student is a debtor, 0 = no")
    tuition_fees_up_to_date: int = Field(..., ge=0, le=1, examples=[1],
        description="1 = fees paid up to date, 0 = not")
    gender: int = Field(..., ge=0, le=1, examples=[1],
        description="1 = male, 0 = female")
    scholarship_holder: int = Field(..., ge=0, le=1, examples=[0],
        description="1 = has scholarship, 0 = no")
    age_at_enrollment: int = Field(..., ge=16, le=80, examples=[20],
        description="Age when enrolled (typical range 17-70)")
    international: int = Field(..., ge=0, le=1, examples=[0],
        description="1 = international student, 0 = no")
    curricular_units_1st_sem_credited: int = Field(..., ge=0, le=30, examples=[0],
        description="Units credited in 1st semester (0-30)")
    curricular_units_1st_sem_enrolled: int = Field(..., ge=0, le=30, examples=[6],
        description="Units enrolled in 1st semester (0-30)")
    curricular_units_1st_sem_evaluations: int = Field(..., ge=0, le=50, examples=[6],
        description="Evaluations taken in 1st semester (0-50)")
    curricular_units_1st_sem_approved: int = Field(..., ge=0, le=30, examples=[6],
        description="Units passed in 1st semester (0-30)")
    curricular_units_1st_sem_grade: float = Field(..., ge=0, le=20, examples=[14.0],
        description="Average grade of 1st semester (0-20)")
    curricular_units_1st_sem_without_evaluations: int = Field(..., ge=0, le=15, examples=[0],
        description="Units without evaluation in 1st semester (0-15)")
    unemployment_rate: float = Field(..., ge=0, le=30, examples=[13.9],
        description="Unemployment rate in % (dataset range ~7.6-16.2)")
    inflation_rate: float = Field(..., ge=-5, le=15, examples=[-0.3],
        description="Inflation rate in % (dataset range ~-0.8 to 3.7)")
    gdp: float = Field(..., ge=-10, le=10, examples=[0.79],
        description="GDP growth (dataset range ~-4.06 to 3.51)")
 
 
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
 
# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
class Reason(BaseModel):
    factor: str
    detail: str
    severity: str  # High / Medium / Low
 
 
class Suggestion(BaseModel):
    category: str  # Financial / Academic / Counselling / Admin
    action: str
    priority: str
 
 
class WhatIf(BaseModel):
    scenario: str
    new_dropout_probability: float
    change: float  # negative = risk goes down
 
 
class PredictionResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "prediction": 0,
        "status": "Not Dropout",
        "dropout_probability": 0.0832,
        "probabilities": {"Dropout": 0.0832, "Not Dropout": 0.9168},
        "risk_level": "Low",
        "confidence": 0.9168,
        "reasons": [{"factor": "Strong academic record",
                     "detail": "Passed 6 of 6 enrolled units with grade 14.0/20.",
                     "severity": "Low"}],
        "suggestions": [{"category": "Academic",
                         "action": "Keep up the current progress; consider peer-mentoring.",
                         "priority": "Low"}],
        "what_if": [],
        "note": None,
    }})
 
    prediction: int
    status: str
    dropout_probability: float
    probabilities: Dict[str, float]
    risk_level: str
    confidence: float
    reasons: List[Reason]
    suggestions: List[Suggestion]
    what_if: List[WhatIf]
    note: Optional[str] = None
 
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_frame(values: dict) -> pd.DataFrame:
    row = {ALIASES[k]: v for k, v in values.items()}
    return pd.DataFrame([row], columns=FEATURES)
 
 
def get_probs(values: dict) -> Dict[str, float]:
    proba = model.predict_proba(to_frame(values))[0]
    return {CLASS_LABELS[i]: float(p) for i, p in enumerate(proba)}
 
 
def dropout_prob(probs: Dict[str, float]) -> float:
    return probs.get("Dropout", 0.0)
 
 
def risk_level(p: float) -> str:
    if p < 0.30:
        return "Low"
    if p < 0.60:
        return "Medium"
    return "High"
 
 
def build_reasons(d: dict) -> List[Reason]:
    r: List[Reason] = []
    enrolled = d["curricular_units_1st_sem_enrolled"]
    approved = d["curricular_units_1st_sem_approved"]
    pass_rate = approved / enrolled if enrolled else 0.0
 
    if enrolled == 0:
        r.append(Reason(factor="No course enrolment",
                        detail="Student enrolled in 0 units in 1st semester.", severity="High"))
    elif pass_rate < 0.5:
        r.append(Reason(factor="Low pass rate",
                        detail=f"Passed only {approved} of {enrolled} units ({pass_rate:.0%}).",
                        severity="High"))
    elif pass_rate < 0.8:
        r.append(Reason(factor="Moderate pass rate",
                        detail=f"Passed {approved} of {enrolled} units ({pass_rate:.0%}).",
                        severity="Medium"))
    else:
        r.append(Reason(factor="Strong academic record",
                        detail=f"Passed {approved} of {enrolled} units with grade "
                               f"{d['curricular_units_1st_sem_grade']}/20.",
                        severity="Low"))
 
    g = d["curricular_units_1st_sem_grade"]
    if enrolled and g < 10:
        r.append(Reason(factor="Low average grade",
                        detail=f"1st semester grade is {g}/20 (below the pass mark of 10).",
                        severity="High"))
    elif enrolled and g < 12:
        r.append(Reason(factor="Borderline grade",
                        detail=f"1st semester grade is {g}/20.", severity="Medium"))
 
    if d["tuition_fees_up_to_date"] == 0:
        r.append(Reason(factor="Unpaid tuition fees",
                        detail="Fees are not up to date - a strong dropout signal.", severity="High"))
    if d["debtor"] == 1:
        r.append(Reason(factor="Student is a debtor",
                        detail="Outstanding debt can force students to leave.", severity="High"))
    if d["scholarship_holder"] == 0 and (d["debtor"] == 1 or d["tuition_fees_up_to_date"] == 0):
        r.append(Reason(factor="No scholarship",
                        detail="Financial stress without scholarship support.", severity="Medium"))
    if d["curricular_units_1st_sem_without_evaluations"] >= 2:
        r.append(Reason(factor="Skipped evaluations",
                        detail=f"{d['curricular_units_1st_sem_without_evaluations']} units had no evaluation - "
                               "sign of disengagement.", severity="Medium"))
    if d["age_at_enrollment"] >= 25:
        r.append(Reason(factor="Mature student",
                        detail=f"Enrolled at age {d['age_at_enrollment']}; work/family load is often higher.",
                        severity="Medium"))
    if d["application_order"] >= 4:
        r.append(Reason(factor="Course was a low-preference choice",
                        detail=f"Application order {d['application_order']} (0 = first choice).",
                        severity="Low"))
    if d["admission_grade"] < 110:
        r.append(Reason(factor="Low admission grade",
                        detail=f"Admission grade {d['admission_grade']}/200.", severity="Medium"))
    if d["displaced"] == 1 and d["scholarship_holder"] == 0:
        r.append(Reason(factor="Living away from home",
                        detail="Displaced student without scholarship: accommodation costs.",
                        severity="Low"))
    if d["daytime_evening_attendance"] == 0:
        r.append(Reason(factor="Evening attendance",
                        detail="Evening students often balance work and study.", severity="Low"))
    if d["unemployment_rate"] > 14:
        r.append(Reason(factor="High unemployment",
                        detail=f"Unemployment rate {d['unemployment_rate']}% adds economic pressure.",
                        severity="Low"))
    return r
 
 
def build_suggestions(d: dict) -> List[Suggestion]:
    s: List[Suggestion] = []
    enrolled = d["curricular_units_1st_sem_enrolled"]
    pass_rate = d["curricular_units_1st_sem_approved"] / enrolled if enrolled else 0.0
 
    if d["tuition_fees_up_to_date"] == 0 or d["debtor"] == 1:
        s.append(Suggestion(category="Financial", priority="High",
                            action="Offer an instalment plan or fee waiver and clear the outstanding balance."))
    if d["scholarship_holder"] == 0 and (d["tuition_fees_up_to_date"] == 0 or d["debtor"] == 1
                                         or d["displaced"] == 1):
        s.append(Suggestion(category="Financial", priority="High",
                            action="Help the student apply for a scholarship, hardship grant or "
                                   "emergency fund."))
    if enrolled and pass_rate < 0.6:
        s.append(Suggestion(category="Academic", priority="High",
                            action="Assign a tutor / remedial classes for the failed units."))
    if enrolled and d["curricular_units_1st_sem_grade"] < 12:
        s.append(Suggestion(category="Academic", priority="Medium",
                            action="Provide study-skills workshops and weekly mentor check-ins."))
    if d["curricular_units_1st_sem_without_evaluations"] >= 2:
        s.append(Suggestion(category="Counselling", priority="Medium",
                            action="Meet the student to find out why evaluations were skipped."))
    if d["age_at_enrollment"] >= 25 or d["daytime_evening_attendance"] == 0:
        s.append(Suggestion(category="Academic", priority="Medium",
                            action="Offer flexible schedules, recorded lectures or part-time load."))
    if d["application_order"] >= 4:
        s.append(Suggestion(category="Counselling", priority="Low",
                            action="Career counselling - confirm the course fits the student's goals "
                                   "or explore a course change."))
    if d["displaced"] == 1:
        s.append(Suggestion(category="Financial", priority="Low",
                            action="Share information on hostel / accommodation support."))
    if not s:
        s.append(Suggestion(category="Academic", priority="Low",
                            action="No major risk found. Keep monitoring and consider peer-mentoring."))
    return s
 
 
# Actions the institution can influence -> used for the what-if simulator
WHAT_IF_SCENARIOS = [
    ("Clear tuition fees (up to date)", {"tuition_fees_up_to_date": 1}),
    ("Remove debt status", {"debtor": 0}),
    ("Award a scholarship", {"scholarship_holder": 1}),
    ("Clear fees + remove debt + scholarship",
     {"tuition_fees_up_to_date": 1, "debtor": 0, "scholarship_holder": 1}),
]
 
 
def build_what_if(d: dict, base: float) -> List[WhatIf]:
    out: List[WhatIf] = []
    for name, changes in WHAT_IF_SCENARIOS:
        if all(d[k] == v for k, v in changes.items()):
            continue  # nothing would change
        new_d = {**d, **changes}
        p = dropout_prob(get_probs(new_d))
        out.append(WhatIf(scenario=name,
                          new_dropout_probability=round(p, 4),
                          change=round(p - base, 4)))
    # Biggest reduction first
    return sorted(out, key=lambda x: x.change)
 
 
# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def home():
    return {"message": "Student Dropout Prediction API is running",
            "docs": "/docs", "sample_input": "/sample"}
 
 
@app.get("/health")
def health():
    return {"status": "healthy", "model_classes": CLASS_LABELS,
            "multiclass": IS_MULTICLASS}
 
 
@app.get("/sample")
def sample():
    """Ready-to-use sample input you can paste into /predict."""
    return SAMPLE_INPUT
 
 
@app.get("/features")
def features():
    """Every input field with its type, min, max and description."""
    props = StudentData.model_json_schema()["properties"]
    return {
        name: {
            "type": p.get("type"),
            "min": p.get("minimum"),
            "max": p.get("maximum"),
            "example": (p.get("examples") or [None])[0],
            "description": p.get("description"),
        }
        for name, p in props.items()
    }
 
 
@app.post("/predict", response_model=PredictionResponse)
def predict(student: StudentData):
    d = student.model_dump()
 
    try:
        probs = get_probs(d)
    except Exception as exc:  # keep API errors readable
        raise HTTPException(status_code=500, detail=f"Model error: {exc}")
 
    top_label = max(probs, key=probs.get)
    p_drop = dropout_prob(probs)
 
    # Treat "Dropout" as prediction 1 to stay compatible with the old API
    prediction = 1 if top_label == "Dropout" else 0
    status = "Dropout" if prediction == 1 else (
        top_label if IS_MULTICLASS else "Not Dropout")
 
    note = None
    if not IS_MULTICLASS:
        note = ("This model is binary (Dropout vs Not Dropout). Retrain with the 3 classes "
                "(Dropout / Enrolled / Graduate) to get Enrolled and Graduate probabilities.")
 
    return PredictionResponse(
        prediction=prediction,
        status=status,
        dropout_probability=round(p_drop, 4),
        probabilities={k: round(v, 4) for k, v in probs.items()},
        risk_level=risk_level(p_drop),
        confidence=round(probs[top_label], 4),
        reasons=build_reasons(d),
        suggestions=build_suggestions(d),
        what_if=build_what_if(d, p_drop),
        note=note,
    )
 
