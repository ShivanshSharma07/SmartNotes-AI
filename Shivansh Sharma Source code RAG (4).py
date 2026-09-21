# ---- RAG PROJECT: AI HANDBOOK ASSISTANT ----
# Beginner-friendly implementation (fixed and improved)

import os
import sys
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


# -------------------------------------------------------
# 1. Load Documents
# -------------------------------------------------------

def load_documents(folder: str) -> Tuple[List[str], List[str]]:
    """Load all .txt files from a folder. Returns list of document texts and filenames."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Data folder not found: {folder}")

    docs: List[str] = []
    file_names: List[str] = []

    for file in sorted(os.listdir(folder)):
        if file.lower().endswith(".txt"):
            path = os.path.join(folder, file)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                docs.append(text)
                file_names.append(file)
    return docs, file_names


# -------------------------------------------------------
# 2. Chunk Documents
# -------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """Split long text into chunks of roughly chunk_size words with an optional overlap.

    chunk_size and overlap are in words (not characters).
    """
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    i = 0
    while i < len(words):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


# -------------------------------------------------------
# 3. Create Embeddings
# -------------------------------------------------------

def embed_chunks(chunks: List[str], model: SentenceTransformer) -> np.ndarray:
    """Return embeddings for a list of chunks as a 2D numpy array."""
    if not chunks:
        return np.zeros((0, model.get_sentence_embedding_dimension()))

    # encode returns a numpy array by default
    embeddings = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    return np.array(embeddings)


# -------------------------------------------------------
# 4. Retrieve Relevant Chunks
# -------------------------------------------------------

def retrieve(query: str, chunk_embeddings: np.ndarray, chunks: List[str], model: SentenceTransformer, top_k: int = 3) -> List[Tuple[str, float]]:
    """Return top_k (chunk, score) pairs for the query. Handles empty index gracefully."""
    if chunk_embeddings.size == 0 or len(chunks) == 0:
        return []

    query_embedding = model.encode([query], convert_to_numpy=True)

    # cosine_similarity expects 2D arrays
    scores = cosine_similarity(query_embedding, chunk_embeddings)[0]

    # get top_k indices (safely)
    top_k = min(top_k, len(scores))
    top_indices = np.argsort(scores)[-top_k:][::-1]

    results = [(chunks[i], float(scores[i])) for i in top_indices]
    return results


# -------------------------------------------------------
# 5. Answer using LLM
# -------------------------------------------------------

def generate_answer(question: str, context: str, generator_pipeline, max_length: int = 150) -> str:
    """Create a prompt from context + question and use a text-generation pipeline to answer.

    The function strips the prompt from the model output when possible.
    """
    # keep context to a reasonable length (characters) to avoid huge prompts
    max_context_chars = 3000
    if len(context) > max_context_chars:
        # prefer the end of context (most recent / relevant sentences)
        context = context[-max_context_chars:]

    prompt = f"Answer the question based on the context.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

    # generate
    output = generator_pipeline(prompt, max_length=len(prompt.split()) + max_length, num_return_sequences=1)
    generated = output[0].get("generated_text", "")

    # If the model returned the prompt + answer, remove the prompt prefix
    if generated.startswith(prompt):
        answer = generated[len(prompt) :].strip()
    else:
        # fallback: remove the prompt text if found anywhere
        answer = generated.replace(prompt, "").strip()

    # safety trim
    if len(answer) > 2000:
        answer = answer[:2000] + "..."

    return answer


# -------------------------------------------------------
# MAIN RAG PIPELINE
# -------------------------------------------------------

def run_rag(data_folder: str = "data"):
    print("Loading documents...")
    try:
        docs, file_names = load_documents(data_folder)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    if not docs:
        print("No documents found in the data folder. Put .txt files into the folder and retry.")
        sys.exit(1)

    # Chunk all documents
    print("Chunking...")
    all_chunks: List[str] = []
    for doc, fname in zip(docs, file_names):
        chunks = chunk_text(doc)
        # optionally prefix chunk with filename to give context about source
        prefixed = [f"[source: {fname}] " + c for c in chunks]
        all_chunks.extend(prefixed)

    # Load embedding model
    print("Loading embedding model (SentenceTransformer all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Create embeddings
    print("Embedding chunks...")
    chunk_embeddings = embed_chunks(all_chunks, embedder)

    # Load a lightweight text-generation model for the answer step
    print("Loading text-generation pipeline (distilgpt2)...")
    generator = pipeline("text-generation", model="distilgpt2", device=-1)

    print("\nRAG System Ready! Ask your questions.\n")

    while True:
        try:
            user_q = input("\nYour Question (type 'exit' to quit): ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if user_q is None:
            continue
        if user_q.strip().lower() == "exit":
            break

        # Step 1: Retrieve relevant chunks
        retrieved = retrieve(user_q, chunk_embeddings, all_chunks, embedder, top_k=3)

        if not retrieved:
            print("No relevant context found in the documents.")
            continue

        top_context = "\n\n".join([f"(score: {score:.4f})\n{c}" for c, score in retrieved])

        print("\n--- Retrieved Context ---")
        print(top_context)

        # Step 2: Generate final answer
        answer = generate_answer(user_q, "\n\n".join([c for c, s in retrieved]), generator)

        print("\n--- Final Answer ---")
        print(answer)
        print("\n----------------------")


# Run the system
if __name__ == "__main__":
    # you can pass a folder path as the first arg: python rag_ai_handbook_assistant.py ./my_data
    folder = sys.argv[1] if len(sys.argv) > 1 else "data"
    run_rag(folder)











