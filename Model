"""
model.py  (v2 — corrected)
──────────────────────────
Rule-based Disease Prediction Model
Converted from Weka J48 Decision Tree (treesJ48 - Training)

All branches verified and corrected against Training_clean.csv.
No retraining performed — purely rule-based logic derived from
Weka tree screenshots and discriminative-feature analysis.

Feature values: binary (0 = absent, 1 = present)
Threshold:      <= 0 → symptom absent  |  > 0 → symptom present
"""

ALL_FEATURES = [
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing",
    "shivering", "chills", "joint_pain", "stomach_pain", "acidity",
    "ulcers_on_tongue", "muscle_wasting", "vomiting", "burning_micturition",
    "spotting_ urination", "fatigue", "weight_gain", "anxiety",
    "cold_hands_and_feets", "mood_swings", "weight_loss", "restlessness",
    "lethargy", "patches_in_throat", "irregular_sugar_level", "cough",
    "high_fever", "sunken_eyes", "breathlessness", "sweating", "dehydration",
    "indigestion", "headache", "yellowish_skin", "dark_urine", "nausea",
    "loss_of_appetite", "pain_behind_the_eyes", "back_pain", "constipation",
    "abdominal_pain", "diarrhoea", "mild_fever", "yellow_urine",
    "yellowing_of_eyes", "acute_liver_failure", "swelling_of_stomach",
    "swelled_lymph_nodes", "malaise", "blurred_and_distorted_vision",
    "phlegm", "throat_irritation", "redness_of_eyes", "sinus_pressure",
    "runny_nose", "congestion", "chest_pain", "weakness_in_limbs",
    "fast_heart_rate", "pain_during_bowel_movements", "pain_in_anal_region",
    "bloody_stool", "irritation_in_anus", "neck_pain", "dizziness",
    "cramps", "bruising", "obesity", "swollen_legs", "swollen_blood_vessels",
    "puffy_face_and_eyes", "enlarged_thyroid", "brittle_nails",
    "swollen_extremeties", "excessive_hunger", "extra_marital_contacts",
    "drying_and_tingling_lips", "slurred_speech", "knee_pain",
    "hip_joint_pain", "muscle_weakness", "stiff_neck", "swelling_joints",
    "movement_stiffness", "spinning_movements", "loss_of_balance",
    "unsteadiness", "weakness_of_one_body_side", "loss_of_smell",
    "bladder_discomfort", "foul_smell_of urine", "continuous_feel_of_urine",
    "passage_of_gases", "internal_itching", "toxic_look_(typhos)",
    "depression", "irritability", "muscle_pain", "altered_sensorium",
    "red_spots_over_body", "belly_pain", "abnormal_menstruation",
    "dischromic _patches", "watering_from_eyes", "increased_appetite",
    "polyuria", "family_history", "mucoid_sputum", "rusty_sputum",
    "lack_of_concentration", "visual_disturbances",
    "receiving_blood_transfusion", "receiving_unsterile_injections",
    "coma", "stomach_bleeding", "distention_of_abdomen",
    "history_of_alcohol_consumption", "fluid_overload",
    "blood_in_sputum", "prominent_veins_on_calf", "palpitations",
    "painful_walking", "pus_filled_pimples", "blackheads", "scurring",
    "skin_peeling", "silver_like_dusting", "small_dents_in_nails",
    "inflammatory_nails", "blister", "red_sore_around_nose",
    "yellow_crust_ooze",
]


def _get(s, feature):
    return s.get(feature, 0)


def predict(symptoms_dict):
    s = symptoms_dict

    # ── A: Thyroid (root node) ─────────────────────────────────────────────
    if _get(s, "abnormal_menstruation") > 0:
        if _get(s, "enlarged_thyroid") <= 0:
            return "Hyperthyroidism"
        return "Hypothyroidism"

    # ── B: Unique discriminative features (checked early) ─────────────────
    if _get(s, "receiving_blood_transfusion") > 0 or _get(s, "receiving_unsterile_injections") > 0:
        return "Hepatitis B"

    if _get(s, "extra_marital_contacts") > 0 and _get(s, "patches_in_throat") > 0:
        return "AIDS"

    if _get(s, "extra_marital_contacts") > 0 and _get(s, "muscle_wasting") > 0:
        return "AIDS"

    if _get(s, "patches_in_throat") > 0 and _get(s, "high_fever") > 0:
        return "AIDS"

    if _get(s, "coma") > 0 and _get(s, "stomach_bleeding") > 0:
        return "Hepatitis E"

    if _get(s, "polyuria") > 0 and _get(s, "increased_appetite") > 0:
        return "Diabetes "

    if _get(s, "blood_in_sputum") > 0:
        return "Tuberculosis"

    if _get(s, "rusty_sputum") > 0:
        return "Pneumonia"

    if _get(s, "palpitations") > 0:
        return "Hypoglycemia"

    if _get(s, "pain_behind_the_eyes") > 0 and _get(s, "muscle_pain") > 0:
        return "Dengue"

    if _get(s, "pain_behind_the_eyes") > 0 and _get(s, "back_pain") > 0:
        return "Dengue"

    if _get(s, "red_spots_over_body") > 0 and _get(s, "malaise") > 0:
        return "Chicken pox"

    if _get(s, "red_sore_around_nose") > 0 or _get(s, "yellow_crust_ooze") > 0:
        return "Impetigo"

    if (_get(s, "runny_nose") > 0 and _get(s, "sinus_pressure") > 0
            and _get(s, "throat_irritation") > 0):
        return "Common Cold"

    if _get(s, "burning_micturition") > 0 and _get(s, "bladder_discomfort") > 0:
        return "Urinary tract infection"

    if _get(s, "foul_smell_of urine") > 0 and _get(s, "continuous_feel_of_urine") > 0:
        return "Urinary tract infection"

    if _get(s, "bladder_discomfort") > 0 and _get(s, "continuous_feel_of_urine") > 0:
        return "Urinary tract infection"

    if _get(s, "burning_micturition") > 0 and _get(s, "spotting_ urination") > 0:
        return "Drug Reaction"

    if _get(s, "spotting_ urination") > 0 and _get(s, "stomach_pain") > 0:
        return "Drug Reaction"

    if _get(s, "burning_micturition") > 0 and _get(s, "stomach_pain") > 0 and _get(s, "skin_rash") > 0:
        return "Drug Reaction"

    if _get(s, "prominent_veins_on_calf") > 0 and _get(s, "bruising") > 0:
        return "Varicose veins"

    if (_get(s, "fluid_overload") > 0
            or (_get(s, "distention_of_abdomen") > 0
                and _get(s, "history_of_alcohol_consumption") > 0)):
        return "Alcoholic hepatitis"

    if _get(s, "mucoid_sputum") > 0 and _get(s, "breathlessness") > 0:
        return "Bronchial Asthma"

    if _get(s, "pain_during_bowel_movements") > 0 and _get(s, "bloody_stool") > 0:
        return "Dimorphic hemmorhoids(piles)"

    if _get(s, "irritation_in_anus") > 0 and _get(s, "pain_in_anal_region") > 0:
        return "Dimorphic hemmorhoids(piles)"

    if _get(s, "movement_stiffness") > 0 and _get(s, "swelling_joints") > 0:
        return "Arthritis"

    if _get(s, "movement_stiffness") > 0 and _get(s, "painful_walking") > 0:
        return "Arthritis"

    if _get(s, "muscle_weakness") > 0 and _get(s, "stiff_neck") > 0 and _get(s, "painful_walking") > 0:
        return "Arthritis"

    if _get(s, "hip_joint_pain") > 0 and _get(s, "knee_pain") > 0:
        return "Osteoarthristis"

    if _get(s, "hip_joint_pain") > 0 and _get(s, "swelling_joints") > 0:
        return "Osteoarthristis"

    if _get(s, "hip_joint_pain") > 0 and _get(s, "painful_walking") > 0:
        return "Osteoarthristis"

    if _get(s, "neck_pain") > 0 and _get(s, "weakness_in_limbs") > 0:
        return "Cervical spondylosis"

    if _get(s, "spinning_movements") > 0 and _get(s, "unsteadiness") > 0:
        return "(vertigo) Paroymsal  Positional Vertigo"

    if _get(s, "lack_of_concentration") > 0 and _get(s, "loss_of_balance") > 0:
        return "Hypertension "

    if _get(s, "lack_of_concentration") > 0 and _get(s, "dizziness") > 0:
        return "Hypertension "

    if _get(s, "lack_of_concentration") > 0 and _get(s, "chest_pain") > 0:
        return "Hypertension "

    if _get(s, "altered_sensorium") > 0 and _get(s, "weakness_of_one_body_side") > 0:
        return "Paralysis (brain hemorrhage)"

    if _get(s, "weakness_of_one_body_side") > 0 and _get(s, "headache") > 0:
        return "Paralysis (brain hemorrhage)"

    if _get(s, "weakness_of_one_body_side") > 0 and _get(s, "vomiting") > 0:
        return "Paralysis (brain hemorrhage)"

    if (_get(s, "breathlessness") > 0 and _get(s, "sweating") > 0
            and _get(s, "chest_pain") > 0):
        return "Heart attack"

    if _get(s, "vomiting") > 0 and _get(s, "sweating") > 0 and _get(s, "chest_pain") > 0:
        return "Heart attack"

    if _get(s, "vomiting") > 0 and _get(s, "breathlessness") > 0 and _get(s, "sweating") > 0:
        return "Heart attack"

    if _get(s, "internal_itching") > 0 and _get(s, "passage_of_gases") > 0:
        return "Peptic ulcer diseae"

    if _get(s, "sunken_eyes") > 0 and _get(s, "dehydration") > 0:
        return "Gastroenteritis"

    if _get(s, "silver_like_dusting") > 0 or _get(s, "small_dents_in_nails") > 0:
        return "Psoriasis"

    if _get(s, "pus_filled_pimples") > 0 and _get(s, "blackheads") > 0:
        return "Acne"

    if _get(s, "blackheads") > 0 and _get(s, "scurring") > 0:
        return "Acne"

    if _get(s, "visual_disturbances") > 0 and _get(s, "acidity") > 0:
        return "Migraine"

    if _get(s, "visual_disturbances") > 0 and _get(s, "stiff_neck") > 0:
        return "Migraine"

    if _get(s, "visual_disturbances") > 0 and _get(s, "irritability") > 0:
        return "Migraine"

    if _get(s, "excessive_hunger") > 0 and _get(s, "stiff_neck") > 0:
        return "Migraine"

    if _get(s, "depression") > 0 and _get(s, "irritability") > 0 and _get(s, "stiff_neck") > 0:
        return "Migraine"

    if _get(s, "shivering") > 0 and _get(s, "watering_from_eyes") > 0:
        return "Allergy"

    if _get(s, "continuous_sneezing") > 0 and _get(s, "watering_from_eyes") > 0:
        return "Allergy"

    if _get(s, "shivering") > 0 and _get(s, "continuous_sneezing") > 0 and _get(s, "chills") > 0:
        return "Allergy"

    if _get(s, "stomach_pain") > 0 and _get(s, "ulcers_on_tongue") > 0:
        return "GERD"

    if _get(s, "acidity") > 0 and _get(s, "vomiting") > 0 and _get(s, "cough") > 0:
        return "GERD"

    if _get(s, "acidity") > 0 and _get(s, "chest_pain") > 0 and _get(s, "cough") > 0:
        return "GERD"

    # ── C: Hepatitis / jaundice cluster ────────────────────────────────────
    if _get(s, "yellowing_of_eyes") > 0 or _get(s, "yellowish_skin") > 0:
        if _get(s, "coma") > 0:
            return "Hepatitis E"
        if (_get(s, "swelling_of_stomach") > 0 and _get(s, "abdominal_pain") > 0
                and (_get(s, "history_of_alcohol_consumption") > 0
                     or _get(s, "fluid_overload") > 0)):
            return "Alcoholic hepatitis"
        if _get(s, "muscle_pain") > 0:
            return "hepatitis A"
        if _get(s, "family_history") > 0:
            return "Hepatitis C"
        if _get(s, "dark_urine") > 0 and _get(s, "joint_pain") > 0:
            return "Hepatitis D"
        if (_get(s, "joint_pain") > 0 and _get(s, "vomiting") > 0
                and _get(s, "fatigue") > 0 and _get(s, "muscle_pain") <= 0):
            return "Hepatitis D"
        if (_get(s, "dark_urine") > 0 and _get(s, "vomiting") > 0
                and _get(s, "muscle_pain") <= 0 and _get(s, "weight_loss") <= 0
                and _get(s, "high_fever") <= 0):
            return "Hepatitis D"
        if (_get(s, "fatigue") > 0 and _get(s, "loss_of_appetite") > 0
                and _get(s, "nausea") > 0 and _get(s, "dark_urine") <= 0):
            return "Hepatitis C"
        if _get(s, "itching") > 0 and _get(s, "nausea") > 0 and _get(s, "dark_urine") <= 0:
            return "Chronic cholestasis"
        if _get(s, "itching") > 0 and _get(s, "abdominal_pain") > 0 and _get(s, "yellowing_of_eyes") > 0:
            return "Chronic cholestasis"
        if (_get(s, "nausea") > 0 and _get(s, "loss_of_appetite") > 0
                and _get(s, "yellowing_of_eyes") > 0 and _get(s, "dark_urine") <= 0
                and _get(s, "weight_loss") <= 0 and _get(s, "muscle_pain") <= 0
                and _get(s, "joint_pain") <= 0 and _get(s, "family_history") <= 0):
            return "Chronic cholestasis"
        if _get(s, "phlegm") > 0 and _get(s, "swelled_lymph_nodes") > 0:
            return "Tuberculosis"
        if _get(s, "weight_loss") > 0 or _get(s, "high_fever") > 0:
            return "Jaundice"
        return "Jaundice"

    # ── D: Weka main trunk ─────────────────────────────────────────────────
    if _get(s, "muscle_pain") > 0:
        if _get(s, "pain_behind_the_eyes") > 0:
            return "Dengue"
        if _get(s, "mild_fever") <= 0:
            return "Malaria"
        return "hepatitis A"

    if _get(s, "malaise") > 0:
        if _get(s, "yellowing_of_eyes") <= 0:
            if _get(s, "chest_pain") <= 0:
                return "Pneumonia"
            if _get(s, "chills") <= 0:
                return "Chicken pox"
            return "Dengue"
        if _get(s, "mild_fever") <= 0:
            if _get(s, "phlegm") <= 0:
                return "Tuberculosis"
            return "Common Cold"
        return "Hepatitis B"

    # ── E: Irritability / chills / weight_loss ─────────────────────────────
    if _get(s, "irritability") > 0:
        if _get(s, "slurred_speech") > 0:
            return "Migraine"
        if _get(s, "chills") > 0:
            if _get(s, "fatigue") <= 0:
                return "Jaundice"
            return "Typhoid"
        return "Hypoglycemia"

    if _get(s, "chills") > 0:
        if _get(s, "slurred_speech") > 0:
            return "Migraine"
        if _get(s, "fatigue") > 0:
            if _get(s, "increased_appetite") > 0:
                return "Diabetes "
            if _get(s, "cough") > 0:
                return "Pneumonia"
            return "Typhoid"
        if _get(s, "increased_appetite") > 0:
            return "Diabetes "
        return "Jaundice"

    if _get(s, "weight_loss") > 0:
        if _get(s, "coma") > 0:
            return "Hepatitis E"
        if _get(s, "itching") > 0:
            return "Jaundice"
        return "Hepatitis D"

    # ── F: Skin/nodal/itching cluster ──────────────────────────────────────
    if _get(s, "skin_peeling") > 0:
        if _get(s, "skin_rash") > 0:
            return "Psoriasis"
        return "Acne"

    if _get(s, "skin_rash") > 0:
        if _get(s, "muscle_wasting") > 0:
            return "AIDS"
        if _get(s, "pus_filled_pimples") > 0:
            return "Acne"
        if _get(s, "joint_pain") > 0:
            return "Psoriasis"
        if _get(s, "high_fever") > 0 and _get(s, "blister") > 0:
            return "Impetigo"
        if _get(s, "nodal_skin_eruptions") > 0:
            if _get(s, "continuous_sneezing") > 0:
                return "Allergy"
            if _get(s, "stomach_pain") > 0:
                return "Drug Reaction"
            return "Fungal infection"
        return "Fungal infection"

    if _get(s, "itching") > 0:
        if _get(s, "nodal_skin_eruptions") > 0:
            if _get(s, "continuous_sneezing") > 0:
                return "Allergy"
            return "Fungal infection"
        return "Fungal infection"

    # ── G: Fatigue / structural diseases ──────────────────────────────────
    if _get(s, "fatigue") > 0:
        if _get(s, "cramps") > 0:
            if _get(s, "obesity") > 0:
                if _get(s, "restlessness") > 0:
                    return "Diabetes "
                return "Varicose veins"
            if _get(s, "cough") > 0:
                return "Bronchial Asthma"
            return "Hepatitis C"
        if _get(s, "obesity") > 0:
            return "Varicose veins"
        if _get(s, "family_history") > 0:
            if _get(s, "high_fever") > 0:
                return "Bronchial Asthma"
            return "Hepatitis C"
        if _get(s, "neck_pain") > 0:
            if _get(s, "stiff_neck") > 0:
                if _get(s, "knee_pain") > 0:
                    return "Osteoarthristis"
                if _get(s, "acidity") > 0:
                    return "Migraine"
                return "Arthritis"
            if _get(s, "knee_pain") > 0:
                return "Osteoarthristis"
            return "Cervical spondylosis"
        if _get(s, "high_fever") > 0:
            if _get(s, "breathlessness") > 0:
                return "Bronchial Asthma"
            return "Typhoid"
        if _get(s, "vomiting") > 0:
            if _get(s, "dehydration") > 0:
                return "Gastroenteritis"
            return "Typhoid"
        return "Typhoid"

    if _get(s, "neck_pain") > 0:
        if _get(s, "knee_pain") > 0:
            return "Osteoarthristis"
        return "Cervical spondylosis"

    if _get(s, "joint_pain") > 0:
        if _get(s, "family_history") > 0:
            if _get(s, "high_fever") > 0:
                return "Bronchial Asthma"
            if _get(s, "cramps") > 0:
                if _get(s, "obesity") > 0:
                    return "Varicose veins"
                return "Hepatitis C"
            return "Varicose veins"
        if _get(s, "vomiting") > 0:
            if _get(s, "high_fever") > 0:
                return "Hepatitis D"
            return "Hepatitis C"
        return "Osteoarthristis"

    # ── H: Slurred speech ──────────────────────────────────────────────────
    if _get(s, "slurred_speech") > 0:
        return "Hypoglycemia"

    # ── I: Vomiting / headache / chest cluster ────────────────────────────
    if _get(s, "vomiting") > 0:
        if _get(s, "sunken_eyes") > 0:
            return "Gastroenteritis"
        if _get(s, "blister") > 0:
            return "Impetigo"
        if _get(s, "headache") > 0:
            if _get(s, "spinning_movements") > 0 or _get(s, "loss_of_balance") > 0:
                return "(vertigo) Paroymsal  Positional Vertigo"
            if _get(s, "altered_sensorium") > 0:
                return "Paralysis (brain hemorrhage)"
            return "(vertigo) Paroymsal  Positional Vertigo"
        if _get(s, "breathlessness") > 0:
            if _get(s, "chest_pain") > 0:
                return "Heart attack"
            return "Bronchial Asthma"
        if _get(s, "abdominal_pain") > 0:
            return "Peptic ulcer diseae"
        return "Gastroenteritis"

    if _get(s, "headache") > 0:
        if _get(s, "loss_of_balance") > 0:
            return "Hypertension "
        if _get(s, "acidity") > 0 or _get(s, "stiff_neck") > 0:
            return "Migraine"
        if _get(s, "back_pain") > 0:
            return "Hypertension "
        return "Migraine"

    if _get(s, "chest_pain") > 0:
        if _get(s, "breathlessness") > 0:
            return "Heart attack"
        if _get(s, "cough") > 0:
            return "Pneumonia"
        return "GERD"

    # ── J: Fallback ────────────────────────────────────────────────────────
    if _get(s, "high_fever") > 0:
        if _get(s, "pain_behind_the_eyes") > 0:
            return "Dengue"
        if _get(s, "mild_fever") > 0:
            return "Malaria"
        return "Typhoid"

    if _get(s, "mild_fever") > 0:
        if _get(s, "phlegm") > 0:
            return "Common Cold"
        return "Malaria"

    if _get(s, "back_pain") > 0:
        return "Cervical spondylosis"

    if _get(s, "acidity") > 0:
        if _get(s, "indigestion") > 0:
            return "GERD"
        return "Migraine"

    if _get(s, "constipation") > 0:
        return "Dimorphic hemmorhoids(piles)"

    if _get(s, "dizziness") > 0:
        return "Hypertension "

    if _get(s, "obesity") > 0:
        return "Varicose veins"

    return "Unknown — please review symptoms"


def predict_from_list(symptom_values):
    if len(symptom_values) != len(ALL_FEATURES):
        raise ValueError(f"Expected {len(ALL_FEATURES)} features, got {len(symptom_values)}")
    return predict(dict(zip(ALL_FEATURES, symptom_values)))


if __name__ == "__main__":
    tests = [
        ({"abnormal_menstruation": 1, "enlarged_thyroid": 1},                                "Hypothyroidism"),
        ({"abnormal_menstruation": 1, "enlarged_thyroid": 0},                                "Hyperthyroidism"),
        ({"muscle_pain": 1, "pain_behind_the_eyes": 1},                                      "Dengue"),
        ({"muscle_pain": 1, "pain_behind_the_eyes": 0, "mild_fever": 0},                     "Malaria"),
        ({"malaise": 1, "yellowing_of_eyes": 0, "chest_pain": 0},                            "Pneumonia"),
        ({"malaise": 1, "yellowing_of_eyes": 0, "chest_pain": 1, "chills": 0},               "Chicken pox"),
        ({"malaise": 1, "yellowing_of_eyes": 0, "chest_pain": 1, "chills": 1},               "Dengue"),
        ({"receiving_blood_transfusion": 1},                                                  "Hepatitis B"),
        ({"extra_marital_contacts": 1, "patches_in_throat": 1},                               "AIDS"),
        ({"coma": 1, "stomach_bleeding": 1},                                                  "Hepatitis E"),
        ({"polyuria": 1, "increased_appetite": 1},                                            "Diabetes "),
        ({"blood_in_sputum": 1},                                                              "Tuberculosis"),
        ({"red_spots_over_body": 1, "malaise": 1},                                            "Chicken pox"),
        ({"red_sore_around_nose": 1},                                                         "Impetigo"),
        ({"runny_nose": 1, "sinus_pressure": 1, "throat_irritation": 1},                     "Common Cold"),
        ({"burning_micturition": 1, "bladder_discomfort": 1},                                 "Urinary tract infection"),
        ({"prominent_veins_on_calf": 1, "bruising": 1},                                       "Varicose veins"),
        ({"fluid_overload": 1},                                                               "Alcoholic hepatitis"),
        ({"mucoid_sputum": 1, "breathlessness": 1},                                          "Bronchial Asthma"),
        ({"pain_during_bowel_movements": 1, "bloody_stool": 1},                               "Dimorphic hemmorhoids(piles)"),
        ({"movement_stiffness": 1, "swelling_joints": 1},                                    "Arthritis"),
        ({"hip_joint_pain": 1, "knee_pain": 1},                                              "Osteoarthristis"),
        ({"neck_pain": 1, "weakness_in_limbs": 1},                                           "Cervical spondylosis"),
        ({"spinning_movements": 1, "unsteadiness": 1},                                       "(vertigo) Paroymsal  Positional Vertigo"),
        ({"lack_of_concentration": 1, "loss_of_balance": 1},                                 "Hypertension "),
        ({"altered_sensorium": 1, "weakness_of_one_body_side": 1},                           "Paralysis (brain hemorrhage)"),
        ({"breathlessness": 1, "sweating": 1, "chest_pain": 1},                              "Heart attack"),
        ({"internal_itching": 1, "passage_of_gases": 1},                                     "Peptic ulcer diseae"),
        ({"sunken_eyes": 1, "dehydration": 1},                                               "Gastroenteritis"),
        ({"silver_like_dusting": 1},                                                          "Psoriasis"),
        ({"pus_filled_pimples": 1, "blackheads": 1},                                         "Acne"),
        ({"visual_disturbances": 1, "acidity": 1},                                           "Migraine"),
        ({"shivering": 1, "watering_from_eyes": 1},                                          "Allergy"),
        ({"stomach_pain": 1, "ulcers_on_tongue": 1},                                         "GERD"),
        ({"yellowing_of_eyes": 1, "dark_urine": 1, "joint_pain": 1},                         "Hepatitis D"),
        ({"yellowing_of_eyes": 1, "family_history": 1},                                      "Hepatitis C"),
        ({"yellowing_of_eyes": 1, "itching": 1, "nausea": 1},                                "Chronic cholestasis"),
    ]
    print("=" * 65)
    print("Rule-based model v2 — quick test suite")
    print("=" * 65)
    passed = 0
    for symptoms, expected in tests:
        result = predict(symptoms)
        status = "✅" if result == expected else "❌"
        if result != expected:
            print(f"  {status}  Expected: {expected:45s}  Got: {result}")
        else:
            passed += 1
    print(f"\n  {passed}/{len(tests)} tests passed")
    print("=" * 65)
