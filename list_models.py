import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

try:
    models = [m.name for m in genai.list_models() if 'embedContent' in m.supported_generation_methods]
    print("AVAILABLE_MODELS: " + ", ".join(models))
except Exception as e:
    print(f"Error: {e}")
