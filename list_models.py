import google.generativeai as genai
import os
import sys

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    # Try reading from iris_permissions or just ask
    print("No API key found in environment.")
    sys.exit(1)

genai.configure(api_key=api_key)
print("Available Models:")
try:
    for m in genai.list_models():
        print(f"Name: {m.name}, DisplayName: {m.display_name}, Methods: {m.supported_generation_methods}")
except Exception as e:
    print(f"Error listing models: {e}")
