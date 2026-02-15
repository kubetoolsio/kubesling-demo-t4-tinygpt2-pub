import os
import time
from fastapi import FastAPI
from pydantic import BaseModel

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = os.getenv("MODEL_ID", "mistralai/Mistral-7B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

app = FastAPI(title="KubeSling HF Mistral-7B Demo")

class InferRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 40

# Loaded once at startup
tokenizer = None
model = None
device = None
startup_t = None

@app.on_event("startup")
def load_model():
    global tokenizer, model, device, startup_t
    t0 = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    token_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **token_kwargs)
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        **token_kwargs,
    )
    model.to(device)
    model.eval()

    startup_t = time.time() - t0

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "startup_seconds": startup_t,
    }

@app.post("/infer")
def infer(req: InferRequest):
    t0 = time.time()
    inputs = tokenizer(req.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=True,
            top_p=0.95,
            temperature=0.9,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return {
        "device": device,
        "latency_ms": int((time.time() - t0) * 1000),
        "output": text,
    }
