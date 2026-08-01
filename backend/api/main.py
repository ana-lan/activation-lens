from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import threading
from model.base_model import load_base_model
from model.evaluate_sae import load_trained_sae
from model.find_top_features import find_top_activating_examples_with_model
from fastapi import HTTPException
from pydantic import BaseModel, Field, field_validator
from safety.live_monitor_multifeature import monitor_generation_multifeature

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
sae.eval()

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

class SafetyMonitorRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    max_new_tokens: int = Field(default=30, ge=1, le=100)

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str):
        value = value.strip()
        if not value:
            raise ValueError("Prompt must not be blank.")
        return value

@app.post("/safety/monitor")
def run_safety_monitor(request: SafetyMonitorRequest):
    prompt_tokens = len(model.tokenizer.encode(request.prompt, add_special_tokens=False)) + 1

    if prompt_tokens + request.max_new_tokens > model.cfg.n_ctx:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Prompt uses {prompt_tokens} tokens; prompt plus {request.max_new_tokens} "
                f"generated tokens exceeds the {model.cfg.n_ctx}-token context window."
            ),
        )

    with inference_lock:
        return monitor_generation_multifeature(
            model, sae, prompt=request.prompt, max_new_tokens=request.max_new_tokens
        )