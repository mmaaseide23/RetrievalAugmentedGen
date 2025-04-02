import os
import pandas as pd
from ChromaIngest import Chroma  # Assumes your Chroma class is in Chroma.py
import ollama
from measure import timer, memory  # Decorators to measure execution time and memory usage

# List of 5 prompts to test (replace these with your actual prompts if needed)
prompts = [
    "What is the main topic of the document?",
    "Summarize the content of the document.",
    "What are the key findings presented?",
    "List the major conclusions from the document.",
    "How does the document describe the methodology?"
]

def generate_rag_response(query, context_results, modelrun):
    """
    Generate a RAG response using the retrieved context and the Ollama mistral model.
    """
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk: {result.get('chunk', 'N/A')}) with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )
    
    prompt = f"""You are a helpful AI assistant.
Use the following context to answer the query as accurately as possible.

Context:
{context_str}

Query: {query}

Answer:"""
    
    response = ollama.chat(model=modelrun, messages=[{"role": "user", "content": prompt}])
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
    for model in models:
        for embedder_name, embedder_param in embedders:
            for config in configs:
                # Build the collection name based on your naming convention
                collection_name = f"awesome_collection_{embedder_name}_{config['chunk_size']}_{config['overlap']}"
                print(f"\n--- Testing Collection: {collection_name} ---")
                
                chroma_instance = Chroma(collection_name=collection_name, embedding_function=embedder_param)
                
                for prompt in prompts:
                    @timer
                    @memory
                    def measure_search():
                        return chroma_instance.get_embedding_chroma(prompt)
                    
                    (context_results, search_memory), search_time = measure_search()
                    
                    @timer
                    @memory
                    def measure_rag():
                        return generate_rag_response(prompt, context_results, model)
                    
                    (rag_result, rag_memory), rag_time = measure_rag()
                    
                    results.append({
                        "model": model,
                        "embedder": embedder_name,
                        "chunk_size": config["chunk_size"],
                        "overlap": config["overlap"],
                        "prompt": prompt,
                        "search_time_ms": search_time,
                        "search_memory_kb": search_memory,
                        "rag_time_ms": rag_time,
                        "rag_memory_kb": rag_memory,
                        "rag_result": rag_result
                    })
                    
                    print(f"Embedder: {embedder_name}, Config: {config}")
                    print(f"Prompt: '{prompt[:30]}...'")
                    print(f"Search - Time: {search_time:.2f} ms, Memory: {search_memory:.2f} KB")
                    print(f"RAG    - Time: {rag_time:.2f} ms, Memory: {rag_memory:.2f} KB\n")
    
    return results

if __name__ == "__main__":
    performance_results = test_rag_performance(prompts)
    
    df = pd.DataFrame(performance_results)



