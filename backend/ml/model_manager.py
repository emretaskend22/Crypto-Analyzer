# ml/model_manager.py

import os
import joblib
from tensorflow.keras.models import load_model as keras_load_model

# Global dict to store loaded models
models = {}

def load_model(model_name: str, model_path: str):
    try:
        ext = os.path.splitext(model_path)[1].lower()
        if ext in [".h5", ".keras"]:
            model = keras_load_model(model_path)
        elif ext in [".pkl", ".joblib"]:
            model = joblib.load(model_path)
        else:
            raise ValueError(f"Unsupported model file extension: {ext}")

        models[model_name] = model
        print(f"[INFO] Loaded '{model_name}' from {model_path}")
        return model
    except Exception as e:
        print(f"[ERROR] load_model('{model_name}') failed: {e}")
        models[model_name] = None
        return None

def get_model(name_or_path: str):
    # Try exact match
    model = models.get(name_or_path)
    if model is not None:
        return model

    # Try matching by filename only
    key = os.path.splitext(os.path.basename(name_or_path))[0]
    return models.get(key)

def load_all_models(models_dir: str):
    if not os.path.exists(models_dir):
        print(f"[WARNING] Models directory '{models_dir}' does not exist.")
        return

    for fname in os.listdir(models_dir):
        path = os.path.join(models_dir, fname)
        if os.path.isfile(path) and fname.lower().endswith((".h5", ".keras", ".pkl", ".joblib")):
            # Extract coin name from filename
            model_name = os.path.splitext(fname)[0]
            load_model(model_name, path)

# ---------------------------
# Load models at startup
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

load_all_models(MODELS_DIR)
