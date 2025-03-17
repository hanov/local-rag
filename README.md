# RAG Query Script

This script builds a FAISS index from text documents and enables question answering (RAG) using either a local LLM (via Ollama) or OpenAI’s API. It splits documents into overlapping chunks, embeds them with Sentence Transformers, and retrieves the most relevant chunks to generate a final answer.

---

## Features

• Text chunking with configurable size and overlap  
• FAISS index creation and loading  
• Retrieval augmented generation with three answer methods ("stuff", "map_reduce", "refine")  
• Configurable to use either a local LLM (Ollama) or OpenAI's API  

---

## Prerequisites

• Python 3.7+ installed

---

## Installation

1. **Clone the repository:**

   Run the following commands in your terminal:
   > git clone <repository_url>
   > cd <repository_folder>

2. **(Optional) Create and activate a virtual environment:**

   • On macOS/Linux:
   > python3 -m venv venv  
   > source venv/bin/activate

   • On Windows:
   > python -m venv venv  
   > venv\Scripts\activate

3. **Install required packages:**

   Run:
   > pip install faiss-cpu numpy sentence-transformers requests openai

   Note: The script sets environment variables to work around OpenMP issues on MacOS (especially for M1/M2).

---

## Configuration

You can configure the script using environment variables. The most important ones are:

• **DOCUMENTS_PATH:** Folder containing your `.txt` files  
• **INDEX_PATH:** Path to save/load the FAISS index  
• **CHUNKS_PATH:** Path to save/load the text chunks  
• **CHUNK_SIZE & CHUNK_OVERLAP:** Parameters for splitting the text  
• **TOP_K:** Number of chunks to retrieve per query  
• **EMBEDDING_MODEL:** SentenceTransformer model name  
• **LOCAL_LLM:** Set "true" to use a local LLM (e.g., Ollama) or "false" for OpenAI  
• **OLLAMA_URL:** URL for the local LLM (default: http://localhost:11434/api/generate)  
• **OLLAMA_MODEL:** Local LLM model name (default: mistral)  
• **OPENAI_API_KEY:** Your OpenAI API key  
• **OPENAI_MODEL:** OpenAI model name (default: gpt-3.5-turbo)  
• **ANSWER_METHOD:** Method for generating the final answer ("stuff", "map_reduce", or "refine")

Example configuration (Unix/macOS):
```
   export DOCUMENTS_PATH="documents"
   export INDEX_PATH="index/faiss_index.index"
   export CHUNKS_PATH="index/chunks.json"
   export CHUNK_SIZE=500
   export CHUNK_OVERLAP=50
   export TOP_K=3
   export EMBEDDING_MODEL="sentence-transformers/distiluse-base-multilingual-cased-v2"
   export LOCAL_LLM="true"
   export OLLAMA_URL="http://localhost:11434/api/generate"
   export OLLAMA_MODEL="mistral"
   export OPENAI_API_KEY="your_openai_api_key_here"
   export OPENAI_MODEL="gpt-3.5-turbo"
   export ANSWER_METHOD="stuff"
```
---

## Usage

1. Place your `.txt` documents in the folder specified by DOCUMENTS_PATH.
2. Run the script:
   > python script.py

3. Follow the on-screen instructions. You will be prompted to enter your question—the script will then retrieve the most relevant context and generate an answer.


---
## Building the Docker Image

To build the Docker image, run the following command in the root of your repository (where the Dockerfile is located):

  docker build -t my-python-app .

This command builds the image and tags it as "my-python-app". You can choose a different tag name if desired.

## Running the Docker Container

After building the image, you can run the container with:

  docker run --rm -it my-python-app

If your application needs to receive environment variables, you can specify them at runtime using the `-e` flag. For example:

  docker run --rm -it -e DOCUMENTS_PATH=documents -e INDEX_PATH=index/faiss_index.index my-python-app

### Mounting Volumes

If your application accesses external data, you may mount directories to the container. For instance, if you need to mount a local directory into the container’s `/app/data` folder, run:

  docker run --rm -it -v /local/path/to/data:/app/data my-python-app

---

## Acknowledgements

This script uses FAISS for vector search, Sentence Transformers for embeddings, and integrates external LLMs for answer generation. Enjoy!
