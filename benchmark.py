from Embedding import InstructorEmbedder, MiniLMEmbedder
from measure import timer, memory
import numpy as np
from typing import List, Tuple, Dict
import time

def generate_test_data(num_chunks: int = 100) -> List[str]:
    """Generate test data with varying complexity"""
    chunks = []
    for i in range(num_chunks):
        if i % 3 == 0:
            chunks.append(f"Short chunk {i}")
        elif i % 3 == 1:
            chunks.append(f"This is a medium length chunk {i} that contains some technical information about machine learning and artificial intelligence.")
        else:
            chunks.append(f"This is a very long chunk {i} that contains detailed technical information about deep learning, neural networks, and various machine learning algorithms. It includes specific details about model architectures, training procedures, and optimization techniques.")
    return chunks

@timer
@memory
def benchmark_embedding(embedder, chunks: List[str], instruction: str = None) -> Tuple[np.ndarray, float, float]:
    """Benchmark the embedding process"""
    if isinstance(embedder, InstructorEmbedder):
        result = embedder.embed_chunks(chunks, instruction)
    else:
        result = embedder.embed_chunks(chunks)
    return result

@timer
@memory
def benchmark_query(embedder, query: str, instruction: str = None) -> Tuple[np.ndarray, float, float]:
    """Benchmark the query embedding process"""
    if isinstance(embedder, InstructorEmbedder):
        result = embedder.embed_query(query, instruction)
    else:
        result = embedder.embed_query(query)
    return result

@timer
def benchmark_similarity_search(embedder, query_embedding: np.ndarray, chunk_embeddings: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray, float]:
    """Benchmark the similarity search process"""
    result = embedder.find_similar_chunks(query_embedding, chunk_embeddings, top_k)
    return result

def evaluate_retrieval_quality(embedder, chunks: List[str], chunk_embeddings: np.ndarray, 
                             query: str, doc_instruction: str = None, query_instruction: str = None, 
                             top_k: int = 5) -> None:
    """Evaluate the qualitative retrieval quality"""
    # Get embeddings
    if isinstance(embedder, InstructorEmbedder):
        query_embedding = embedder.embed_query(query, query_instruction)
    else:
        query_embedding = embedder.embed_query(query)
    
    # Find similar chunks
    top_indices, scores = embedder.find_similar_chunks(query_embedding, chunk_embeddings, top_k)
    
    print(f"\nRetrieval Quality Assessment for Query: '{query}'")
    print("-" * 80)
    for idx, score in zip(top_indices, scores):
        print(f"Score: {score:.4f}")
        print(f"Retrieved Chunk: {chunks[idx]}")
        print("-" * 80)

def benchmark_model(embedder, test_chunks: List[str], test_queries: List[str], 
                   doc_instruction: str = None, query_instruction: str = None) -> Dict:
    """Run complete benchmark for a single model"""
    model_name = embedder.__class__.__name__
    print(f"\n{'='*20} Benchmarking {model_name} {'='*20}")
    
    # Benchmark embedding process
    print("\nBenchmarking document embedding process...")
    (chunk_embeddings, memory_used), embedding_time = benchmark_embedding(
        embedder, test_chunks, doc_instruction
    )
    print(f"Document Embedding Time: {embedding_time:.2f} ms")
    print(f"Document Embedding Memory: {memory_used:.2f} KB")
    
    # Benchmark query process
    print("\nBenchmarking query embedding process...")
    query_times = []
    query_memories = []
    search_times = []
    
    for query in test_queries:
        (query_embedding, memory_used), query_time = benchmark_query(
            embedder, query, query_instruction
        )
        query_times.append(query_time)
        query_memories.append(memory_used)
        
        # Benchmark similarity search
        (indices, scores), search_time = benchmark_similarity_search(
            embedder, query_embedding, chunk_embeddings
        )
        search_times.append(search_time)
        
        print(f"\nQuery: '{query}'")
        print(f"Query Embedding Time: {query_time:.2f} ms")
        print(f"Query Embedding Memory: {memory_used:.2f} KB")
        print(f"Similarity Search Time: {search_time:.2f} ms")
    
    # Print average metrics
    print(f"\n{model_name} Average Performance Metrics:")
    metrics = {
        'avg_query_time': np.mean(query_times),
        'avg_query_memory': np.mean(query_memories),
        'avg_search_time': np.mean(search_times),
        'doc_embedding_time': embedding_time,
        'doc_embedding_memory': memory_used
    }
    
    print(f"Average Query Embedding Time: {metrics['avg_query_time']:.2f} ms")
    print(f"Average Query Embedding Memory: {metrics['avg_query_memory']:.2f} KB")
    print(f"Average Search Time: {metrics['avg_search_time']:.2f} ms")
    
    # Evaluate retrieval quality
    print(f"\nEvaluating {model_name} Retrieval Quality...")
    for query in test_queries:
        evaluate_retrieval_quality(
            embedder, test_chunks, chunk_embeddings, query, 
            doc_instruction, query_instruction
        )
    
    return metrics

def main():
    # Initialize embedders
    instructor_embedder = InstructorEmbedder()
    minilm_embedder = MiniLMEmbedder()
    
    # Generate test data
    print("\nGenerating test data...")
    test_chunks = generate_test_data(100)
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "Explain neural networks",
        "How does deep learning work?",
        "What are optimization techniques?",
        "Tell me about model architectures"
    ]
    
    # Instructions (only for Instructor)
    doc_instruction = "Represent the document for retrieval:"
    query_instruction = "Represent the question for retrieving supporting documents:"
    
    # Benchmark InstructorEmbedder
    instructor_metrics = benchmark_model(
        instructor_embedder, test_chunks, test_queries,
        doc_instruction, query_instruction
    )
    
    # Benchmark MiniLMEmbedder
    minilm_metrics = benchmark_model(
        minilm_embedder, test_chunks, test_queries
    )
    
    # Compare models
    print("\n" + "="*50)
    print("Model Comparison Summary")
    print("="*50)
    metrics = {
        'Document Embedding Time (ms)': ('doc_embedding_time', 'lower'),
        'Document Embedding Memory (KB)': ('doc_embedding_memory', 'lower'),
        'Average Query Time (ms)': ('avg_query_time', 'lower'),
        'Average Query Memory (KB)': ('avg_query_memory', 'lower'),
        'Average Search Time (ms)': ('avg_search_time', 'lower')
    }
    
    for metric_name, (metric_key, better) in metrics.items():
        instructor_value = instructor_metrics[metric_key]
        minilm_value = minilm_metrics[metric_key]
        winner = "InstructorEmbedder" if (instructor_value < minilm_value) == (better == 'lower') else "MiniLMEmbedder"
        
        print(f"\n{metric_name}:")
        print(f"InstructorEmbedder: {instructor_value:.2f}")
        print(f"MiniLMEmbedder: {minilm_value:.2f}")
        print(f"Winner: {winner}")

if __name__ == "__main__":
    main() 