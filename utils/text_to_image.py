import os
import requests
import io
from PIL import Image

API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"

def get_headers():
    token = os.getenv("HF_TOKEN")
    if not token:
        return None
    return {"Authorization": f"Bearer {token}"}

def generate_image_from_text(prompt):
    headers = get_headers()
    if not headers:
        return None, "HF_TOKEN not found in environment variables."
    
    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json={
                "inputs": prompt,
                "parameters": {"num_inference_steps": 4}
            },
            timeout=60,
        )
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content)), None
        elif response.status_code == 503:
            return None, "Model is starting up. Try again in 20 seconds."
        else:
            return None, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return None, f"Connection Error: {e}"
