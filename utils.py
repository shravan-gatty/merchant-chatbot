import os
from dotenv import load_dotenv
import pandas as pd
import faiss
import numpy as np
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Debug: safely print API key status
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print("Loaded API Key:", api_key[:10], "********")
else:
    print("❌ OPENAI_API_KEY is not loaded. Check your .env file!")

# Initialize OpenAI client with your API key
client = OpenAI(api_key=api_key)

# Step 1: Load data from all CSVs
def load_data():
    chunks = []
    for file in ["data/Support Data(Sheet1).csv", "data/txn_refunds.csv", "data/settlement_data.csv"]:
        try:
            df = pd.read_csv(file)
        except UnicodeDecodeError:
            df = pd.read_csv(file, encoding='latin1')  # fallback encoding
        for _, row in df.iterrows():
            text = f"{file} | " + " | ".join(f"{col}: {row[col]}" for col in df.columns)
            chunks.append(text)
    return chunks

# Step 2: Convert text to embeddings using OpenAI
def get_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# Step 3: Create FAISS index
def build_index(chunks):
    dim = 1536  # embedding dimension for ada-002
    index = faiss.IndexFlatL2(dim)
    embeddings = [get_embedding(chunk) for chunk in chunks]
    index.add(np.array(embeddings).astype('float32'))
    return index, chunks

# Step 4: Search top relevant chunks for a query
def search_chunks(index, query, chunks, k=5):
    query_embed = get_embedding(query)
    D, I = index.search(np.array([query_embed]).astype('float32'), k)
    return [chunks[i] for i in I[0]]

# Step 5: Ask GPT for an answer based on context
def ask_gpt(question, context):
    context_text = "\n".join(context)
    prompt = f"""Use the below data to answer the user query.

Data:
{context_text}

Question: {question}
Answer:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
