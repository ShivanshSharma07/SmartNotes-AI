# SmartNotes AI

SmartNotes AI is a simple Retrieval-Augmented Generation (RAG) project built with Python. It retrieves relevant information from text documents and generates answers based on the retrieved context.

## Features

- Load text documents
- Split documents into smaller chunks
- Generate text embeddings
- Retrieve relevant information using cosine similarity
- Generate answers using an AI text-generation model

## Tech Stack

- Python
- NumPy
- Sentence Transformers
- Scikit-learn
- Hugging Face Transformers

## How It Works

1. Load documents from the data folder
2. Split documents into chunks
3. Create embeddings for the chunks
4. Find relevant chunks for the user's question
5. Generate an answer using the retrieved context

## How to Run

```bash
pip install -r requirements.txt
python rag_ai_handbook_assistant.py
