from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import threading
from model.base_model import load_base_model
from model.evaluate_sae import load_trained_sae
from model.find_top_features import find_top_activating_examples_with_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cpu"
model = load_base_model()
model = model.to(device)

sae = load_trained_sae("model/sae_weights.pt")
sae = sae.to(device)

inference_lock = threading.Lock()

@app.get("/features/{feature_idx}")
def get_feature_examples(feature_idx: int, top_k: int = 10):
    with inference_lock:
        examples = find_top_activating_examples_with_model(model, sae, feature_idx=feature_idx, top_k=top_k)

    results = [
        {"activation": val, "context": context}
        for val, context in examples
    ]

    return {"feature_idx": feature_idx, "examples": results}