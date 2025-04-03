import os
import pandas as pd
from ChromaIngest import Chroma  
import ollama
from measure import timer, memory 


embedder = None
chroma_instance = Chroma(
    collection_name=f"awesome_collection_MiniLM_1000_200",
    embedding_function=embedder)

prompts = [
    "Write a Mongo query based on the movies data set that returns the titles of all movies released between 2010 and 2015 from the suspense genre.",
    "Why is a B+ Tree a better than an AVL tree when indexing a large dataset?",
    "Add 23 to the AVL Tree below.  What imbalance case is created with inserting 23?30/\\ 2535/20",
    "When was Redis originally released?",
    "Insert the key:value pairs sequentially into an initially empty AVL tree: "
]

def generate_rag_response(query, context_results):
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
    
    response = ollama.chat(model="tinyllama:latest", messages=[{"role": "user", "content": prompt_text}])
    return response["message"]["content"]

def interactive_search():
    """Interactive search interface."""
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        raw_context_result = chroma_instance.get_embedding_chroma(query)

        docs = raw_context_result.get("documents", [])
        metadatas = raw_context_result.get("metadatas", [])
        distances = raw_context_result.get("distances", [0.0] * len(docs))
        processed_context = []
        for doc, meta, distance in zip(docs, metadatas, distances):
            # Since metadata is stored as a list with a single dictionary, extract the first element.
            meta_item = meta[0] if isinstance(meta, list) and meta else {}
            # Ensure the similarity is a single number
            similarity_value = distance[0] if isinstance(distance, list) and distance else distance
            processed_context.append({
                "file": meta_item.get("file", "Unknown file"),
                "page": meta_item.get("page", "Unknown page"),
                "chunk": doc,
                "similarity": similarity_value
            })

        # Print retrieved context
        print("\n--- Retrieved Context ---")
        for result in processed_context:
            print(f"File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}")

        # Generate RAG response using the retrieved context
        response = generate_rag_response(query, processed_context)

        # Print the generated response
        print("\n--- Response ---")
        print(response)


    
   

if __name__ == "__main__":
    performance_results = interactive_search()
    
