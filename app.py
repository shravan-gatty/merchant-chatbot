from flask import Flask, request, jsonify
from dotenv import load_dotenv
from utils import load_data, build_index, search_chunks, ask_gpt

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)

# Load data and build index on startup
print("Loading payment data and building index...")
chunks = load_data()
index, chunk_texts = build_index(chunks)
print("Index ready!")

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "Query is missing"}), 400

    try:
        top_chunks = search_chunks(index, query, chunk_texts)
        answer = ask_gpt(query, top_chunks)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "✅ Merchant Insights Chatbot is running!"

# ✅ Required change for Render to bind to the correct host/port
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Render will provide PORT
    app.run(host="0.0.0.0", port=port)
