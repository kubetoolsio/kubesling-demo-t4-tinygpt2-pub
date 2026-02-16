# KubeSling Demo: Mistral-7B on GPU

Health: GET /health
Infer: POST /infer { "prompt": "...", "max_new_tokens": 40 }

MODEL_ID default: mistralai/Mistral-7B-Instruct-v0.3 (override via env var if needed)
HF_TOKEN: set this env var with a Hugging Face access token if the chosen model is gated.
