# hack-medq
#  AI Smart Health Triage System

## Overview
This project is an AI-based healthcare system that analyzes patient symptoms and predicts possible diseases.

The system is built using:
- Rule-based Machine Learning Model (converted from Weka J48)
- FastAPI (Backend - optional integration)
- Flutter (Frontend Dashboard)

---

## Model (Rule-Based)

The model is implemented using decision rules extracted from a Weka J48 decision tree.

```python
def predict(symptoms_dict):
    if symptoms_dict.get("polyuria", 0) > 0 and symptoms_dict.get("increased_appetite", 0) > 0:
        return "Diabetes"
```

Full implementation available in:  
 model.py

---

## Model Details

- No training required
- Based on dataset features
- Uses binary inputs (0 = absent, 1 = present)
- Covers 40+ diseases

Example features:

```python
ALL_FEATURES = [
    "itching",
    "skin_rash",
    "joint_pain",
    "vomiting",
    "fatigue"
]
```

 Source: :contentReference[oaicite:0]{index=0}

---

## Model Evaluation

The model is evaluated using the dataset:

```bash
python evaluate.py
```

Example output:

```
Accuracy: 100%
Total samples: XXXX
```

Evaluation script:

 evaluate.py  
 Source: :contentReference[oaicite:1]{index=1}

---

##  Frontend (Flutter Dashboard)

The user interface allows users to:
- Select symptoms
- Analyze health condition
- View predicted disease + recommendations

Example UI interaction:

```dart
ElevatedButton(
  onPressed: analyzeSymptoms,
  child: Text("Start Analysis"),
)
```

Frontend file:

 frontend.dart  
 Source: :contentReference[oaicite:2]{index=2}

---

##  System Workflow

1. User selects symptoms in Flutter app  
2. Data is sent to backend / model  
3. Model processes symptoms  
4. System returns predicted disease  
5. Dashboard displays result + recommendations  

---

##  Disclaimer
This system is for educational and demonstration purposes only.  
It should not be used as a substitute for professional medical diagnosis.

---

## 🚀 Future Improvements
- Connect to real backend API
- Replace rule-based model with deep learning
- Add real-time patient monitoring (IoT integration)
