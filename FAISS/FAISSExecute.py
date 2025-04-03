import numpy as np
import ollama
from Embedding import MiniLMEmbedder, MPNetEmbedder, InstructorEmbedder
from FAISSIngest import FAISS

# Embedders
embedders = {
    "Instructor": InstructorEmbedder(),
    "MiniLM": MiniLMEmbedder(),
    "MPNet": MPNetEmbedder(),
}

# Pipeline Selections
# Chunk/Overlap
config = {"chunk_size": 500, "overlap": 100}

# LLM
llm_model = "llama3.2:latest" 

# Embedder (Same name from embedders above)
embedder_name = "MiniLM"
embedder = embedders[embedder_name]

# Load FAISS database
collection_name = f"faiss_data_{embedder_name}_{config['chunk_size']}_{config['overlap']}"
print(f"Loading FAISS model: {collection_name}")
faiss_db = FAISS(collection_name=f"awesome_collection_{embedder_name}_{config['chunk_size']}_{config['overlap']}", 
                 embedding_function=embedder.model.encode)
faiss_db.load(collection_name)

def search_embeddings(query, top_k=3):
    """Search FAISS for similar embeddings."""
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
        print(f"Search error: {e}")
        return []

def generate_rag_response(query, context_results):
    """Generate a response using retrieved context and LLM."""
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
            f"with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )

    prompt = f"""You are a helpful AI assistant. 
    Use the following context to answer the query accurately. If the context is 
    not relevant, say 'I don't know'. If there's not enough information, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""

    response = ollama.chat(model=llm_model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

def interactive_search():
    """Interactive search interface."""
    print(f"🔍 RAG Search Interface (Using {llm_model} with {embedder_name} Embeddings)")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        # Retrieve relevant embeddings
        context_results = search_embeddings(query)

        # Print retrieved context
        print("\n--- Retrieved Context ---")
        for result in context_results:
            print(f"File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}")

        # Generate response
        response = generate_rag_response(query, context_results)

        # Print the generated response
        print("\n--- Response ---")
        print(response)

if __name__ == "__main__":
    interactive_search()
