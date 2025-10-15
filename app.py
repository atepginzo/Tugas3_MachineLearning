import os
import json
import numpy as np
from flask import Flask, render_template, request
from joblib import load
import csv


app = Flask(__name__)

# Resolve artifact paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "logistic_model.joblib")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.joblib")
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "feature_list.json")

# Load artifacts at startup
model = load(MODEL_PATH)
scaler = load(SCALER_PATH)
with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    feature_list = json.load(f)

# Determine feature sets
FULL_FEATURES = list(feature_list)  # Full vector expected by the Logistic Regression model
if hasattr(scaler, "feature_names_in_") and getattr(scaler, "feature_names_in_", None) is not None:
    SCALER_FEATURES = list(scaler.feature_names_in_)
else:
    SCALER_FEATURES = FULL_FEATURES

# Preset fields (match form names)
PRESET_FIELDS = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]


def load_presets(csv_path, max_rows=30):
    presets = []
    if not os.path.exists(csv_path):
        return presets
    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= max_rows:
                    break
                # Build a dict containing only fields we need; coerce to strings for form values
                preset = {}
                for k in PRESET_FIELDS:
                    if k in row:
                        preset[k] = row[k]
                # Add an optional label from dataset if available
                label = row.get('default.payment.next.month') or row.get('default', '')
                preset['__label__'] = label
                # Include ID if present for display purposes
                preset['__id__'] = row.get('ID', str(i+1))
                presets.append(preset)
    except Exception:
        # Fail silently, presets remain empty
        return []
    return presets


# Load dataset presets at startup
CSV_PATH = os.path.join(BASE_DIR, "UCI_Credit_Card.csv")
presets = load_presets(CSV_PATH, max_rows=30)


def build_feature_vector(form_data):
    """Build ordered numeric vector matching feature_list from form inputs.

    Expects original inputs including categorical raw values; produces one-hot
    columns that exist in feature_list (e.g., SEX_2, EDUCATION_2/3/4, MARRIAGE_1/2/3).
    Missing features default to 0.
    """

    # Initialize all features with 0 based on full feature set
    features = {name: 0.0 for name in FULL_FEATURES}

    # Continuous / ordinal numeric fields
    numeric_fields = [
        "LIMIT_BAL",
        "AGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ]

    for key in numeric_fields:
        val = form_data.get(key)
        try:
            features[key] = float(val) if val is not None and val != "" else 0.0
        except ValueError:
            features[key] = 0.0

    # Categorical encodings (one-hot)
    # SEX: original 1=male, 2=female; feature_list contains only SEX_2 (female)
    sex = form_data.get("SEX")
    if sex is not None and str(sex) == "2":
        if "SEX_2" in features:
            features["SEX_2"] = 1.0

    # EDUCATION: original values {1,2,3,4,...}; we have dummies 2,3,4 with 1 as baseline
    edu = form_data.get("EDUCATION")
    if edu is not None:
        for cat in ("2", "3", "4"):
            col = f"EDUCATION_{cat}"
            if col in features:
                features[col] = 1.0 if str(edu) == cat else 0.0

    # MARRIAGE: original values {1=married, 2=single, 3=others}; dummies for 1,2,3 (0 would be baseline if present)
    marriage = form_data.get("MARRIAGE")
    if marriage is not None:
        for cat in ("1", "2", "3"):
            col = f"MARRIAGE_{cat}"
            if col in features:
                features[col] = 1.0 if str(marriage) == cat else 0.0

    # Order full features (model expects this order)
    ordered_full = np.array([features[name] for name in FULL_FEATURES], dtype=float).reshape(1, -1)
    return ordered_full


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_label = None
    default_probability = None

    if request.method == "POST":
        # Build full feature vector (27)
        X_full = build_feature_vector(request.form)

        # If scaler was trained on a subset (e.g., numeric columns), scale only those and merge back
        if SCALER_FEATURES and len(SCALER_FEATURES) > 0:
            # Map indices of scaler features within FULL_FEATURES
            idx_map = [FULL_FEATURES.index(col) for col in SCALER_FEATURES if col in FULL_FEATURES]
            # Extract subset to scale
            X_subset = X_full[:, idx_map]
            X_subset_scaled = scaler.transform(X_subset)
            # Put back scaled values into the full vector
            for j, col_idx in enumerate(idx_map):
                X_full[0, col_idx] = X_subset_scaled[0, j]
        X_scaled_full = X_full

        # Predict
        y_pred = model.predict(X_scaled_full)[0]
        proba = getattr(model, "predict_proba", None)
        if callable(proba):
            proba_vals = model.predict_proba(X_scaled_full)[0]
            # Assuming positive class (default) is labeled as 1
            default_probability = float(proba_vals[1])
        else:
            default_probability = None

        prediction_label = "Berisiko" if int(y_pred) == 1 else "Tidak Berisiko"

    return render_template(
        "index.html",
        prediction_label=prediction_label,
        default_probability=default_probability,
        presets=presets,
    )


if __name__ == "__main__":
    # Local development
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)