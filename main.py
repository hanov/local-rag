#!/usr/bin/env python3
import os
# ────────────────────────────────────────────────────────────── 
# Workaround for OpenMP issues on MacOS (especially on M1/M2)
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
# ──────────────────────────────────────────────────────────────

import sys
import json
import glob
import faiss
import numpy as np
import requests
import multiprocessing  # added to set start method

# Set the multiprocessing start method to "spawn"
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    # It may already be set; ignore the error.
    pass

from typing import List
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# CONFIGURATION - environment variables
# --------------------------------------------------
LOCAL_LLM = os.environ.get("LOCAL_LLM", "true").lower() == "true"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")

DOCUMENTS_PATH = os.environ.get("DOCUMENTS_PATH", "documents")
INDEX_PATH = os.environ.get("INDEX_PATH", "index/faiss_index.index")
CHUNKS_PATH = os.environ.get("CHUNKS_PATH", "index/chunks.json")

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "50"))
TOP_K = int(os.environ.get("TOP_K", "3"))

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/distiluse-base-multilingual-cased-v2")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")

# CHOOSE the final answer method: "stuff", "map_reduce", or "refine"
ANSWER_METHOD = os.environ.get("ANSWER_METHOD", "stuff").lower()

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def load_documents(folder_path: str) -> List[str]:
    docs = []
    files = glob.glob(os.path.join(folder_path, "*.txt"))
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read()
            docs.append(text)
    return docs

def build_faiss_index(chunks: List[str], embedding_model_name: str, index_path: str):
    print(f"[INFO] Embedding {len(chunks)} chunks with model: {embedding_model_name}")
    model = SentenceTransformer(embedding_model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    embeddings = embeddings.astype('float32')

    dimension = embeddings.shape[1]
    print(f"[INFO] Creating Faiss index (dimension={dimension})")
    index_flat = faiss.IndexFlatL2(dimension)
    faiss_index = faiss.IndexIDMap2(index_flat)

    ids = np.arange(len(chunks)).astype('int64')
    faiss_index.add_with_ids(embeddings, ids)

    print(f"[INFO] Saving Faiss index to: {index_path}")
    faiss.write_index(faiss_index, index_path)

def load_faiss_index(index_path: str):
    print(f"[INFO] Loading Faiss index from: {index_path}")
    return faiss.read_index(index_path)

def retrieve_top_k(query: str, index, all_chunks: List[str], embedding_model_name: str, top_k: int):
    model = SentenceTransformer(embedding_model_name)
    query_emb = model.encode([query], show_progress_bar=False)
    query_emb = query_emb.astype('float32')

    distances, indices = index.search(query_emb, top_k)

    retrieved_texts = []
    for i in range(top_k):
        doc_id = indices[0][i]
        if doc_id == -1:
            continue
        chunk_text_val = all_chunks[doc_id]
        retrieved_texts.append(chunk_text_val)
    return retrieved_texts

# --------------------------------------------------
# LLM CALLS
# --------------------------------------------------
def call_llm_local_ollama(prompt: str, model: str = "mistral") -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except Exception as e:
        return f"[ERROR] Ollama request failed: {e}"

def call_llm_openai(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    import openai
    openai.api_key = OPENAI_API_KEY
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[ERROR] OpenAI request failed: {e}"

def ask_model(prompt: str) -> str:
    if LOCAL_LLM:
        return call_llm_local_ollama(prompt, model=OLLAMA_MODEL)
    else:
        if not OPENAI_API_KEY:
            return "[ERROR] No OPENAI_API_KEY provided."
        return call_llm_openai(prompt, model=OPENAI_MODEL)

# --------------------------------------------------
# ANSWER METHODS
# --------------------------------------------------
def generate_stuff_answer(main_question: str, knowledge_context: List[str]) -> str:
    combined_context = "\n\n".join(knowledge_context)
    prompt = f"""
You are an AI assistant. The user asked:
\"\"\"{main_question}\"\"\"

Below is all relevant context (in chunks):
{combined_context}

Using ONLY the above context, provide a concise final answer.
If you are unsure, say "I am not sure."

Final answer:
"""
    return ask_model(prompt)

def generate_map_reduce_answer(main_question: str, knowledge_context: List[str]) -> str:
    partial_summaries = []
    for idx, chunk_text_val in enumerate(knowledge_context, start=1):
        map_prompt = f"""
Summarize the following excerpt briefly:
Excerpt #{idx}:
\"\"\"{chunk_text_val}\"\"\"

Brief summary:
"""
        summary = ask_model(map_prompt)
        partial_summaries.append(summary)

    combined_summaries = "\n\n".join(partial_summaries)
    reduce_prompt = f"""
We have brief summaries for several excerpts:

{combined_summaries}

Now, based on these summaries, answer the following question:
\"\"\"{main_question}\"\"\"

Please provide a detailed and precise answer:
"""
    final_answer = ask_model(reduce_prompt)
    return final_answer

def generate_refine_answer(main_question: str, knowledge_context: List[str]) -> str:
    draft_answer = generate_stuff_answer(main_question, knowledge_context)
    refine_prompt = f"""
Here is the preliminary answer:
\"\"\"{draft_answer}\"\"\"

Here is the entire context:
{json.dumps(knowledge_context, ensure_ascii=False, indent=2)}

Review the answer and let us know if anything needs to be clarified or added to make it more complete.
If no changes are needed, please respond with "The answer is already complete."
"""
    refine_response = ask_model(refine_prompt)

    if "The answer is already complete" in refine_response:
        return draft_answer
    else:
        return f"{draft_answer}\n\nAddition:\n{refine_response}"

# --------------------------------------------------
# DIRECT RAG QUERY (BYPASSING RESEARCH STATE)
# --------------------------------------------------
def direct_rag_query(question: str, faiss_index, all_chunks: List[str]) -> str:
    # Retrieve the top-K relevant chunks directly
    retrieved_chunks = retrieve_top_k(question, faiss_index, all_chunks, EMBEDDING_MODEL, top_k=TOP_K)
    # Use the selected answer method to generate the final answer using the retrieved context
    if ANSWER_METHOD == "map_reduce":
        return generate_map_reduce_answer(question, retrieved_chunks)
    elif ANSWER_METHOD == "refine":
        return generate_refine_answer(question, retrieved_chunks)
    else:
        return generate_stuff_answer(question, retrieved_chunks)

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    if not os.path.exists(os.path.dirname(INDEX_PATH)):
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        print("[INFO] Loading existing index...")
        faiss_index = load_faiss_index(INDEX_PATH)
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
    else:
        print("[INFO] Building new index from documents folder:", DOCUMENTS_PATH)
        docs = load_documents(DOCUMENTS_PATH)
        all_chunks = []
        for doc_text in docs:
            doc_chunks = chunk_text(doc_text, CHUNK_SIZE, CHUNK_OVERLAP)
            all_chunks.extend(doc_chunks)

        print(f"[INFO] Total chunks: {len(all_chunks)}")
        if not all_chunks:
            print("[ERROR] No chunks found. Exiting.")
            sys.exit(1)

        build_faiss_index(all_chunks, EMBEDDING_MODEL, INDEX_PATH)
        faiss_index = load_faiss_index(INDEX_PATH)

        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print("\n====================================================")
    print("DIRECT RAG QUERY - READY FOR QUERY")
    print(f"Answer method: {ANSWER_METHOD.upper()}")
    print("Enter your question. Press Enter.")
    print("====================================================\n")
    user_question = sys.stdin.readline().strip()
    if not user_question:
        print("[ERROR] No question entered. Exiting.")
        sys.exit(1)

    final_answer = direct_rag_query(user_question, faiss_index, all_chunks)
    print("\n[FINAL ANSWER]")
    print(final_answer)
    print()

if __name__ == "__main__":
    main()
