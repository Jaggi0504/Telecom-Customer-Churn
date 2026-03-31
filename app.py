from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
import numpy as np
from xgboost import XGBClassifier
import requests
from fastapi.templating import Jinja2Templates


app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Load model
model = XGBClassifier()
model.load_model("xgb_model.json")

# Feature order (IMPORTANT)
FEATURES = [
    "gender","SeniorCitizen","Partner","Dependents","tenure",
    "PhoneService","MultipleLines","OnlineSecurity","OnlineBackup",
    "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
    "PaperlessBilling","MonthlyCharges","TotalCharges",
    "InternetService_DSL","InternetService_Fiber optic","InternetService_No",
    "Contract_Month-to-month","Contract_One year","Contract_Two year",
    "PaymentMethod_Bank transfer (automatic)",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check"
]

# Convert helpers
def bool_map(val):
    return 1 if val == "True" else 0

def gender_map(val):
    return 1 if val == "Male" else 0


@app.get("/", response_class=HTMLResponse)
def form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict(
    gender: str = Form(...),
    SeniorCitizen: str = Form(...),
    Partner: str = Form(...),
    Dependents: str = Form(...),
    tenure: float = Form(...),
    PhoneService: str = Form(...),
    MultipleLines: str = Form(...),
    OnlineSecurity: str = Form(...),
    OnlineBackup: str = Form(...),
    DeviceProtection: str = Form(...),
    TechSupport: str = Form(...),
    StreamingTV: str = Form(...),
    StreamingMovies: str = Form(...),
    PaperlessBilling: str = Form(...),
    MonthlyCharges: float = Form(...),
    TotalCharges: float = Form(...),
    InternetService: str = Form(...),
    Contract: str = Form(...),
    PaymentMethod: str = Form(...)
):

    # Convert inputs
    data = {
        "gender": gender_map(gender),
        "SeniorCitizen": bool_map(SeniorCitizen),
        "Partner": bool_map(Partner),
        "Dependents": bool_map(Dependents),
        "tenure": tenure,
        "PhoneService": bool_map(PhoneService),
        "MultipleLines": bool_map(MultipleLines),
        "OnlineSecurity": bool_map(OnlineSecurity),
        "OnlineBackup": bool_map(OnlineBackup),
        "DeviceProtection": bool_map(DeviceProtection),
        "TechSupport": bool_map(TechSupport),
        "StreamingTV": bool_map(StreamingTV),
        "StreamingMovies": bool_map(StreamingMovies),
        "PaperlessBilling": bool_map(PaperlessBilling),
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
        "InternetService_DSL": 1 if InternetService == "DSL" else 0,
        "InternetService_Fiber optic": 1 if InternetService == "Fiber optic" else 0,
        "InternetService_No": 1 if InternetService == "No" else 0,
        "Contract_Month-to-month": 1 if Contract == "Month-to-month" else 0,
        "Contract_One year": 1 if Contract == "One year" else 0,
        "Contract_Two year": 1 if Contract == "Two year" else 0,
        "PaymentMethod_Bank transfer (automatic)": 1 if PaymentMethod == "Bank transfer" else 0,
        "PaymentMethod_Credit card (automatic)": 1 if PaymentMethod == "Credit card" else 0,
        "PaymentMethod_Electronic check": 1 if PaymentMethod == "Electronic check" else 0,
        "PaymentMethod_Mailed check": 1 if PaymentMethod == "Mailed check" else 0,
    }

    # Arrange in correct order
    features = np.array([data[f] for f in FEATURES]).reshape(1, -1)

    pred = model.predict(features)[0]
    #prob = model.predict_proba(features)[0][1]

    return {
        "Churn Prediction": int(pred)
        #"Probability": float(prob)
    }