import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv

# ── Load environment variables ──────────────────────────
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Load the FAQ knowledge base ─────────────────────────
def load_faq():
    with open("faq.txt", "r", encoding="utf-8") as f:
        return f.read()

# ── Build the system prompt ──────────────────────────────
def build_system_prompt(faq_content):
    return f"""
You are a helpful and professional assistant for [Company Name], 
a consultancy firm specializing in Agriculture, Economics, and Climate services.

Your job is to answer customer questions based ONLY on the company knowledge 
base provided below. Follow these rules strictly:

1. Only answer questions related to the company's services.
2. If the answer is clearly in the knowledge base, answer it helpfully.
3. If the question is not covered in the knowledge base, respond with:
   "I don't have that information. Please contact our team directly for assistance."
4. Do not make up information or answer questions outside the company's services.
5. Keep your answers clear, friendly, and professional.
6. Do not mention that you are an AI or that you are using a knowledge base.

COMPANY KNOWLEDGE BASE:
{faq_content}
"""

# ── Flask app setup ──────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Chat endpoint ────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        faq_content = load_faq()
        system_prompt = build_system_prompt(faq_content)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",   # free and fast model on Groq
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message}
            ],
            temperature=0.3,    # lower = more focused, less creative
            max_tokens=512      # enough for a clear answer
        )

        reply = response.choices[0].message.content
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Health check endpoint ────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Chatbot backend is running"})

# ── Run the server ───────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)