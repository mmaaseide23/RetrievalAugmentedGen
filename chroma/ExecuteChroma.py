import json
import numpy as np
from Embedding import MiniLMEmbedder, MPNetEmbedder, InstructorEmbedder
import ollama
from FAISSIngest import FAISS

# Initialize MiniLM Embedder
embedder = MiniLMEmbedder()

# Initialize FAISS with the embedding function
faiss_db = FAISS(collection_name="awesome_collection", embedding_function=embedder.model.encode)
faiss_db.load("faiss_data")

def get_embedding(text):
    """Generate embedding"""
    return embedder.embed_chunks([text])[0]

def search_embeddings(query, top_k=3):
    """Search the FAISS database for similar embeddings."""
    try:
        results = faiss_db.get_embedding_faiss(query)

        # Transform results into a structured format
        top_results = [
            {
                "file": meta["file"],
                "page": meta["page"],
                "chunk": chunk,
                "similarity": distance,
            }
            for chunk, meta, distance in zip(
                results["documents"],
                results["metadatas"],
                results["distances"]
            )
        ][:top_k]

        # Debugging output
        for result in top_results:
            print(f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}")

        return top_results

    except Exception as e:
        print(f"Search error: {e}")
        return []

def generate_rag_response(query, context_results):
    """Generate a response using the retrieved context and the Ollama Mistral model."""
    
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
            f"with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )

    prompt = f"""You are a helpful AI assistant. 
    Use the following context to answer the query as accurately as possible. If the context is 
    not relevant to the query, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""

    # Generate response using Ollama
    response = ollama.chat(
        model="mistral:latest", messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

def interactive_search():
    """Interactive search interface."""
    print("🔍 FAISS RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        # Search for relevant embeddings
        context_results = search_embeddings(query)

        # Print retrieved context
        print("\n--- Retrieved Context ---")
        # for result in context_results:
        #     print(f"File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}")

        # Generate RAG response using the retrieved context
        response = generate_rag_response(query, context_results)

        # Print the generated response
        print("\n--- Response ---")
        print(response)

if __name__ == "__main__":
    interactive_search() 