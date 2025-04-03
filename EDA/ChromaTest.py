import os
import pandas as pd
from chroma.ChromaIngest import Chroma  # Assumes your Chroma class is in Chroma.py
import ollama
from measure import timer, memory  # Decorators to measure execution time and memory usage

prompts = [
    "Write a Mongo query based on the movies data set that returns the titles of all movies released between 2010 and 2015 from the suspense genre.",
    "Why is a B+ Tree a better than an AVL tree when indexing a large dataset?",
    "Add 23 to the AVL Tree below.  What imbalance case is created with inserting 23?30/\\ 2535/20",
    "When was Redis originally released?",
    "Insert the key:value pairs sequentially into an initially empty AVL tree: "
]

def generate_rag_response(query, context_results, modelrun):
    """
    Generate a RAG response using the retrieved context and the Ollama model.
    """
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk: {result.get('chunk', 'N/A')}) with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )
    
    prompt_text = f"""You are a helpful AI assistant.
Use the following context to answer the query as accurately as possible.

Context:
{context_str}

Query: {query}

Answer:"""
    
    response = ollama.chat(model=modelrun, messages=[{"role": "user", "content": prompt_text}])
    return response["message"]["content"]

def test_rag_performance(prompts):
    """
    Iterates through every collection and prompt.
    Measures and records the performance of both the Chroma search and the RAG generation.
    """
    embedders = [
        ("Instructor", "hkunlp/instructor-xl"),
        ("MiniLM", None),
        ("MPNet", "all-mpnet-base-v2")
    ]
    
    configs = [
        {"chunk_size": 200, "overlap": 50},
        {"chunk_size": 500, "overlap": 100},
        {"chunk_size": 1000, "overlap": 200}
    ]
    
    results = []
    models = ["llama3.2:latest", "tinyllama:latest", "mistral:latest"]
    
    for embedder_name, embedder_param in embedders:
        for config in configs:
            collection_name = f"awesome_collection_{embedder_name}_{config['chunk_size']}_{config['overlap']}"
            print(f"\n--- Testing Collection: {collection_name} ---")
            
            chroma_instance = Chroma(collection_name=collection_name, embedding_function=embedder_param)
            
            for number, prompt in enumerate(prompts):
                @timer
                @memory
                def measure_search():
                    return chroma_instance.get_embedding_chroma(prompt)
                
                (raw_context_result, search_memory), search_time = measure_search()
                

                

                
                results.append({
                    "DB": "Chroma",
                    "Embedder": embedder_name,
                    "Chunk Size": config["chunk_size"],
                    "Overlap": config["overlap"],
                    "Question Number": number,
                    "Context Retrieval Time": search_time,
                    "Context Retrieval Memory": search_memory,
                })
                
                print(f"Embedder: {embedder_name}, Config: {config}")
                print(f"Prompt: '{prompt[:30]}...'")
                print(f"Search - Time: {search_time:.2f} ms, Memory: {search_memory:.2f} KB")

    
    return results

if __name__ == "__main__":
    performance_results = test_rag_performance(prompts)
    
    df = pd.DataFrame(performance_results)
    df.to_csv("chroma_rag_performance.csv", index=False)
    print("Performance metrics saved to chroma_rag_performance.csv")
