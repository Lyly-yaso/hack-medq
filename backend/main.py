from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
import os
import json
import re

from model import predict, ALL_FEATURES

load_dotenv()

app = FastAPI(title="Rafeeq Backend")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


class PredictRequest(BaseModel):
    symptoms: list[str]


class TriageRequest(BaseModel):
    age: int
    weight: float
    gender: str | None = None
    blood_sugar: float | None = None
    cholesterol: float | None = None
    hemoglobin: float | None = None
    blood_pressure: str | None = None
    selected_symptoms: list[str] = []
    chat_messages: list[str] = []


SYMPTOM_KEY_MAP = {
    "spotting_urination": "spotting_ urination",
    "foul_smell_of_urine": "foul_smell_of urine",
    "dischromic_patches": "dischromic _patches",
    "toxic_look": "toxic_look_(typhos)",
}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(data: PredictRequest):
    symptoms_dict = {feature: 0 for feature in ALL_FEATURES}

    for symptom in data.symptoms:
        mapped = SYMPTOM_KEY_MAP.get(symptom, symptom)
        if mapped in symptoms_dict:
            symptoms_dict[mapped] = 1

    disease = predict(symptoms_dict)

    return {
        "disease": disease,
        "selected_symptoms": data.symptoms
    }


def clean_json_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    return text.strip()


def extract_symptoms_from_chat(chat_messages: list[str]) -> list[str]:
    all_keys = ", ".join(ALL_FEATURES)
    joined_chat = "\n".join(chat_messages)

    prompt = f"""
You are a medical symptom extraction assistant.

Your task:
- Read the user's chat messages
- Extract ONLY symptoms that exist in this allowed symptom list
- Return ONLY valid JSON
- Do not explain anything
- Output format must be exactly:

{{
  "symptoms": ["itching", "skin_rash"]
}}

Allowed symptom keys:
{all_keys}

User chat:
{joined_chat}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    raw_text = clean_json_text(response.text)

    try:
        data = json.loads(raw_text)
        symptoms = data.get("symptoms", [])
        fixed = []

        for s in symptoms:
            mapped = SYMPTOM_KEY_MAP.get(s, s)
            if mapped in ALL_FEATURES:
                fixed.append(mapped)

        return list(set(fixed))
    except Exception:
        return []


def generate_health_plan(
    disease: str,
    final_symptoms: list[str],
    data: TriageRequest
) -> dict:
    prompt = f"""
You are an AI healthcare companion.

The diagnosis was already predicted by a decision tree model.
You must NOT change the diagnosis.
You only need to explain it in a simple, patient-friendly way.

Return ONLY valid JSON in this exact format:

{{
  "disease": "same disease name",
  "confidence": 0.90,
  "risk": "Low",
  "top_symptoms": ["symptom1", "symptom2"],
  "diet_recommendations": ["item1", "item2", "item3"],
  "activity_recommendations": ["item1", "item2"],
  "general_habits": ["item1", "item2", "item3"],
  "alert": "short warning/disclaimer text",
  "summary": "short patient-friendly explanation"
}}

Rules:
- Keep disease exactly as: {disease}
- confidence must be between 0.70 and 0.99
- risk must be only one of: Low, Medium, High
- top_symptoms should come from the detected symptoms
- diet_recommendations must be short and practical
- activity_recommendations must be short and practical
- general_habits must be short and practical
- alert must be one short sentence
- summary must be short and simple
- Return JSON only, no markdown

Patient basic info:
- age: {data.age}
- weight: {data.weight}
- gender: {data.gender}
- blood_sugar: {data.blood_sugar}
- cholesterol: {data.cholesterol}
- hemoglobin: {data.hemoglobin}
- blood_pressure: {data.blood_pressure}

Detected symptoms:
{final_symptoms}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    raw_text = clean_json_text(response.text)

    try:
        result = json.loads(raw_text)

        confidence = float(result.get("confidence", 0.85))
        confidence = max(0.70, min(confidence, 0.99))

        risk = result.get("risk", "Medium")
        if risk not in ["Low", "Medium", "High"]:
            risk = "Medium"

        top_symptoms = result.get("top_symptoms", final_symptoms[:5])
        if not isinstance(top_symptoms, list):
            top_symptoms = final_symptoms[:5]

        return {
            "disease": result.get("disease", disease),
            "confidence": confidence,
            "risk": risk,
            "top_symptoms": top_symptoms,
            "diet_recommendations": result.get("diet_recommendations", []),
            "activity_recommendations": result.get("activity_recommendations", []),
            "general_habits": result.get("general_habits", []),
            "alert": result.get("alert", "This is not a final medical diagnosis."),
            "summary": result.get("summary", f"Preliminary result suggests {disease}.")
        }

    except Exception:
        return {
            "disease": disease,
            "confidence": 0.85,
            "risk": "Medium",
            "top_symptoms": final_symptoms[:5],
            "diet_recommendations": [
                "Maintain a balanced diet",
                "Drink enough water",
                "Reduce processed foods"
            ],
            "activity_recommendations": [
                "Walk daily for 20–30 minutes",
                "Do light physical activity regularly"
            ],
            "general_habits": [
                "Sleep well",
                "Monitor symptoms",
                "Seek medical advice if symptoms worsen"
            ],
            "alert": "This is not a final medical diagnosis.",
            "summary": f"Preliminary result suggests {disease}."
        }


@app.post("/triage")
def triage_endpoint(data: TriageRequest):
    chat_symptoms = extract_symptoms_from_chat(data.chat_messages)

    selected_fixed = []
    for s in data.selected_symptoms:
        mapped = SYMPTOM_KEY_MAP.get(s, s)
        if mapped in ALL_FEATURES:
            selected_fixed.append(mapped)

    final_symptoms = list(set(selected_fixed + chat_symptoms))

    symptoms_dict = {feature: 0 for feature in ALL_FEATURES}
    for symptom in final_symptoms:
        symptoms_dict[symptom] = 1

    disease = predict(symptoms_dict)
    plan = generate_health_plan(disease, final_symptoms, data)

    return {
        "disease": plan["disease"],
        "confidence": plan["confidence"],
        "risk": plan["risk"],
        "top_symptoms": plan["top_symptoms"],
        "diet_recommendations": plan["diet_recommendations"],
        "activity_recommendations": plan["activity_recommendations"],
        "general_habits": plan["general_habits"],
        "alert": plan["alert"],
        "summary": plan["summary"],
        "selected_symptoms": selected_fixed,
        "chat_symptoms": chat_symptoms,
        "final_symptoms": final_symptoms,
        "basic_info": {
            "age": data.age,
            "weight": data.weight,
            "gender": data.gender,
            "blood_sugar": data.blood_sugar,
            "cholesterol": data.cholesterol,
            "hemoglobin": data.hemoglobin,
            "blood_pressure": data.blood_pressure
        }
    }