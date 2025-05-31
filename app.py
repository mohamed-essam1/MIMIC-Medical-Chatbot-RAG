from flask import Flask, request, jsonify
import faiss
import numpy as np
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import re
from gensim.utils import simple_preprocess
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
import tensorflow as tf
import gc
from numba import cuda
import json
import os
import logging

# Directory to store downloaded and extracted data
DATA_DIR = Path("./mimic_textbooks")
# Define file path for the saved FAISS index
INDEX_FILE_PATH = "faiss_index.idx"
CHUNKED_DOCS_PATH = "chunked_documents.json"


def save_chunked_documents(documents, file_path=CHUNKED_DOCS_PATH):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(documents, f)

# Function to load chunked documents
def load_chunked_documents(file_path=CHUNKED_DOCS_PATH):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Download and extract the dataset zip file
def download_and_extract_zip(url, extract_to=DATA_DIR):
    extract_to.mkdir(parents=True, exist_ok=True)
    zip_path = extract_to / "textbooks.zip"
    response = requests.get(url, stream=True)
    with open(zip_path, "wb") as file:
        for chunk in tqdm(response.iter_content(chunk_size=1024), unit='KB'):
            if chunk:
                file.write(chunk)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

# Text processing functions
def load_text_files(directory):
    texts = []
    for file_path in Path(directory).glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            texts.append(file.read())
    return texts

def clean_and_tokenize(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = simple_preprocess(text)
    return ' '.join(tokens)

def chunk_text(text, chunk_size=200):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Function to generate embeddings for all chunks in batches with GPU acceleration
def get_embeddings_in_batch(texts, batch_size=16):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Tokenize and move batch to GPU
        inputs = retrieval_tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")

        with torch.no_grad():
            outputs = retrieval_model(**inputs)
            batch_embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()

        all_embeddings.extend(batch_embeddings)
    return np.array(all_embeddings)

# Retrieval function
def get_query_embedding(query):
    inputs = retrieval_tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = retrieval_model(**inputs)
        embedding = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
    return embedding

def retrieve_documents(query, top_k=5):
    query_embedding = get_query_embedding(query).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    return [chunked_documents[idx] for idx in indices[0]]

# Generation function
def generate_response(query, context, gpu_device, max_new_tokens=500):
    input_text = f"User query: {query}\n\nContext:\n{context}\n\nAnswer:"
    inputs = generation_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(gpu_device)
    with torch.no_grad():
        outputs = generation_model.generate(inputs["input_ids"], max_new_tokens=max_new_tokens)
    return generation_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Clear GPU function

def clear_gpu():
  torch.cuda.empty_cache()  # Clear GPU memory from torch
  gc.collect()
  numba_device = cuda.get_current_device() # Clear GPU memory from tf
  numba_device.reset()


# Initialize Flask app
app = Flask(__name__)

print("Clearing GPU...")
clear_gpu()

logging.basicConfig(level=logging.DEBUG)
gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Embedding and FAISS index setup
retrieval_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
retrieval_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(gpu_device)

logging.info("Retrieval model in GPU")

# Check if chunked documents exist, otherwise process and save them
if os.path.exists(CHUNKED_DOCS_PATH):
    logging.info("Loading chunked documents...")
    chunked_documents = load_chunked_documents()
else:
    logging.info("Downloading dataset...")

    # Download and extract textbooks
    dataset_url = "https://www.dropbox.com/scl/fi/54p9kkx5n93bffyx08eba/textbooks.zip?rlkey=2y2c5x8y0uncnddichn9cmd7n&st=m290nmkk&dl=1"
    download_and_extract_zip(dataset_url)

    logging.info("Creating chunks...")

    # Load, clean, and chunk documents
    documents = load_text_files(DATA_DIR / "textbooks/en")
    cleaned_documents = [clean_and_tokenize(doc) for doc in documents]
    chunked_documents = []
    for doc in cleaned_documents:
        chunked_documents.extend(chunk_text(doc))
    save_chunked_documents(chunked_documents)

# Check if FAISS index file exists, otherwise create a new one
if os.path.exists(INDEX_FILE_PATH):
    logging.info("Loading FAISS index from disk...")
    index = faiss.read_index(INDEX_FILE_PATH)
else:

    logging.info("Generating embeddings...")

    # Generate embeddings and populate FAISS index
    embeddings = get_embeddings_in_batch(chunked_documents, batch_size=128)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype("float32"))
    faiss.write_index(index, INDEX_FILE_PATH)

# Clear GPU
# Move model to CPU and clear GPU memory
retrieval_model.to("cpu")
clear_gpu()

logging.info("Retrieval model in CPU, clearing GPU")

# Load the generative model on GPU
generation_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct", trust_remote_code=True)
generation_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-mini-instruct", trust_remote_code=True)
generation_model.to(gpu_device)

logging.info("Generation model in GPU")

print("Initialization complete.")

# Flask endpoint
@app.route("/chat", methods=["POST"])
def chat():
    try:
        logging.info("Received request")
        data = request.get_json()
        user_query = data.get("query")
        max_tokens = data.get("max_tokens", 500)

        logging.info("Retrieving documents")
        retrieved_docs = retrieve_documents(user_query)
        retrieved_text = " ".join(retrieved_docs)

        logging.info("Generating response")
        response_text = generate_response(user_query, retrieved_text, gpu_device=gpu_device, max_new_tokens=max_tokens)

        logging.info("Sending response back")
        return jsonify({"response": response_text})
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return jsonify({"error": "An internal error occurred"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
