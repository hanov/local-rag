# Dockerfile

FROM python:3.10-slim

# Install system dependencies for building Python packages and OpenMP support.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    libomp-dev \
  && rm -rf /var/lib/apt/lists/*

# Set the working directory.
WORKDIR /app

# Copy the current directory contents into the container at /app.
COPY . /app

# Upgrade pip.
RUN pip install --upgrade pip

# Install required Python packages.
RUN pip install faiss-cpu numpy sentence-transformers requests openai

# Define environment variables as needed. You can override these at runtime.
# For example:
# ENV DOCUMENTS_PATH=documents
# ENV INDEX_PATH=index/faiss_index.index

# Specify the default command to run the script.
CMD ["python", "main.py"]
