from InstructorEmbedding import INSTRUCTOR
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class InstructorEmbedder:
    def __init__(self):
        """Initialize the DocumentEmbedder with the InstructorXL model"""
        self.model = INSTRUCTOR('hkunlp/instructor-xl')
        
    def embed_chunks(self, chunks, instruction):
        """Embed a list of text chunks using the INSTRUCTOR model"""
        embedding_inputs = [[instruction, chunk] for chunk in chunks]
        return self.model.encode(embedding_inputs)
    
    def embed_query(self, query, instruction):
        """Embed a query using the instructor model"""
        return self.model.encode([[instruction, query]])
    
    def find_similar_chunks(self, query_embedding, chunk_embeddings, top_k=5):
        """Find the most similar chunks to a query using cosine similarity"""
        similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_scores = similarities[top_indices]
        return top_indices, top_scores

class MiniLMEmbedder:
    def __init__(self):
        """Initialize the DocumentEmbedder with the MiniLM model"""
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed_chunks(self, chunks):
        """Embed a list of text chunks using the MiniLM model"""
        return self.model.encode(chunks)
    
    def embed_query(self, query, instruction=None):  # instruction parameter kept for API compatibility
        """Embed a query using the MiniLM model
        Note: instruction parameter is ignored as MiniLM doesn't use instructions"""
        return self.model.encode([query])
    
    def find_similar_chunks(self, query_embedding, chunk_embeddings, top_k=5):
        """Find the most similar chunks to a query using cosine similarity"""
        similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_scores = similarities[top_indices]
        return top_indices, top_scores

if __name__ == "__main__":
    # Initialize the embedder
    embedder = InstructorEmbedder()
    
    # Example course note chunks
    test_chunks = [
        "Vector databases are specialized databases designed to store and retrieve high-dimensional vectors efficiently.",
        "Embedding models convert text into numerical vectors that capture semantic meaning.",
        "Chunking strategies involve breaking down large documents into smaller, meaningful segments.",
        "RAG (Retrieval Augmented Generation) combines information retrieval with language model generation.",
        "Vector similarity search uses distance metrics like cosine similarity to find related content."
    ]
    
    # Instruction for document embedding
    doc_instruction = "Represent the document for retrieval:"
    
    # Embed the chunks
    print("\nEmbedding chunks...")
    chunk_embeddings = embedder.embed_chunks(test_chunks, doc_instruction)
    print(f"Shape of chunk embeddings: {chunk_embeddings.shape}")
    
    # Test queries
    test_queries = [
        "How do vector databases work?",
        "What is RAG?",
        "Explain embedding models"
    ]
    
    # Query instruction
    query_instruction = "Represent the question for retrieving supporting documents:"
    
    # Test each query
    print("\nTesting queries...")
    for query in test_queries:
        print(f"\nQuery: {query}")
        
        # Embed the query
        query_embedding = embedder.embed_query(query, query_instruction)
        
        # Find similar chunks
        top_indices, scores = embedder.find_similar_chunks(query_embedding, chunk_embeddings, top_k=2)
        
        # Print results
        print("Top 2 most relevant chunks:")
        for idx, score in zip(top_indices, scores):
            print(f"Score: {score:.4f} | Chunk: {test_chunks[idx]}")