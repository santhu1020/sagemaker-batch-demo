import json
import os
import joblib
import numpy as np


def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model", "model.joblib")
    model = joblib.load(model_path)
    return model


def input_fn(request_body, content_type):
    if content_type == "application/json":
        lines = request_body.strip().split("\n")
        data = [json.loads(line)["features"] for line in lines]
        return np.array(data)
    raise ValueError("Unsupported content type")


def predict_fn(data, model):
    return model.predict(data)


def output_fn(predictions, accept):
    if accept == "application/json":
        return "\n".join(
            json.dumps({"prediction": int(p)}) for p in predictions
        ), accept
    raise ValueError("Unsupported accept type")
