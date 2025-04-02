import json
import numpy as np
import pandas as pd
import ollama
import os
from Embedding import MiniLMEmbedder, MPNetEmbedder, InstructorEmbedder
from FAISSIngest import FAISS
from measure import timer, memory
import csv

# Define the prompts
prompts = [
    "Write a Mongo query based on the movies data set that returns the titles of all movies released between 2010 and 2015 from the suspense genre.",
    "Why is a B+ Tree better than an AVL tree when indexing a large dataset?",
    r"Add 23 to the AVL Tree below. What imbalance case is created with inserting 23? 30 /  \ 25  35 / 20",
    "When was Redis originally released?",
    "Insert the key:value pairs sequentially into an initially empty AVL tree: [(20:O), (40:S), (60:T), (80:R), (89:N), (70:E)]"
]

# Define embedders to test
embedders = [
    ("Instructor", InstructorEmbedder()),
    ("MiniLM", MiniLMEmbedder()),
    ("MPNet", MPNetEmbedder())
]

# Define configurations to test
configs = [
    {"chunk_size": 200, "overlap": 50},
    {"chunk_size": 500, "overlap": 100},
    {"chunk_size": 1000, "overlap": 200},
]

# Define LLM models to test
llm_models = ["llama3.2:latest", "tinyllama:latest"]

def get_embedding(text, embedder):
    """Generate embedding using the given embedder"""
    return embedder.embed_chunks([text])[0]

def search_embeddings(query, faiss_db, top_k=3):
    """Search the FAISS database for similar embeddings."""
    try:
        results = faiss_db.get_embedding_faiss(query)
        return [
            {
                "file": meta["file"],
                "page": meta["page"],
                "chunk": chunk,
                "similarity": distance,
            }
            for chunk, meta, distance in zip(results["documents"], results["metadatas"], results["distances"])
        ][:top_k]
    except Exception as e:
        print(f"Search error: {str(e)}")
        return []

@timer
@memory
def generate_rag_response(query, context_results, llm_model):
    """Generate a response using the retrieved context and the given LLM model."""
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
            f"with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )
    prompt = f"""You are a helpful AI assistant. 
    Use the following context to answer the query as accurately as possible. If you are not confident
    about the answer, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""
    response = ollama.chat(model=llm_model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

def log_results(entry, log_file="faiss_rag_results.json"):
    """Log a single result entry to a JSON file."""
    try:
        if not os.path.exists(log_file) or os.stat(log_file).st_size == 0:
            with open(log_file, "w") as f:
                json.dump([], f)
        with open(log_file, "r+") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
            data.append(entry)
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()
    except Exception as e:
        print(f"Error writing to log file: {e}")

def run_faiss_queries(faiss_db, embedder, model_name, chunk_size, overlap, all_results):
    """Run all prompts through FAISS and test different LLMs."""
    for i, prompt in enumerate(prompts, start=1):
        context_results = search_embeddings(prompt, faiss_db)
        for llm in llm_models:
            (response, execution_memory), execution_time = generate_rag_response(prompt, context_results, llm)
            result_entry = {
                "embedder": model_name,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "question_number": i,
                "llm_model": llm,
                "context_retrieval_time_ms": execution_time,
                "context_retrieval_memory": execution_memory,
                "response": response
            }
            log_results(result_entry)
            all_results.append(result_entry)

def main():
    all_results = []
    for model_name, embedder in embedders:
        for config in configs:
            chunk_size = config["chunk_size"]
            overlap = config["overlap"]
            collection_name = f"faiss_data_{model_name}_{chunk_size}_{overlap}"
            print(f"Loading FAISS model: {collection_name}")
            faiss_db = FAISS(collection_name=f"awesome_collection_{model_name}_{chunk_size}_{overlap}", embedding_function=embedder.model.encode)
            faiss_db.load(collection_name)
            run_faiss_queries(faiss_db, embedder, model_name, chunk_size, overlap, all_results)
    df = pd.DataFrame(all_results)
    csv_filename = "faiss_results_all.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")

if __name__ == "__main__":
    main()
