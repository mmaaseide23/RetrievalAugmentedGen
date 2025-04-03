import pandas as pd
import numpy as np
import fitz
import os
import redis
from Embedding import MiniLMEmbedder, MPNetEmbedder, InstructorEmbedder
from measure import timer, memory
from redis.commands.search.query import Query


# Redis client
redis_client = redis.Redis(host="localhost", port=6380, db=0)

VECTOR_DIM = 384  
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"



class DocumentProcessor:
    def __init__(self, embedding_model):
        """Initialize the document processor with the embedding model"""
        self.embedder = embedding_model
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text chunk"""
        return self.embedder.embed_chunks([text])[0]

    def store_embedding(self, file: str, page: str, chunk: str, embedding: np.ndarray):
        """Store document chunk and its embedding in Redis"""
        key = f"{DOC_PREFIX}{file}:{page}:{hash(chunk)}"
        
        embedding_bytes = embedding.astype(np.float32).tobytes()
        
        redis_client.hset(
            key,
            mapping={
                "text": chunk,
                "file": file,
                "page": page,
                "embedding": embedding_bytes
            }
        )
        return "Processing Complete"

    @timer
    @memory
    def process_pdfs(self, data_dir: str, chunking_size, overlap_size):
        """Process all PDFs in the given directory"""
        for file_name in os.listdir(data_dir):
            if file_name.endswith(".pdf"):
                pdf_path = os.path.join(data_dir, file_name)
                text_by_page = self.extract_text_from_pdf(pdf_path)
                for page_num, text in text_by_page:
                    chunks = self.split_text_into_chunks(text, chunking_size, overlap_size)
                    for chunk in chunks:
                        # Get embedding for chunk
                        embedding = self.get_embedding(chunk)
                        # Store in Redis
                        self.store_embedding(
                            file=file_name,
                            page=str(page_num),
                            chunk=chunk,
                            embedding=embedding
                        )
                print(f" -----> Processed {file_name}")
        return "Processing Complete"

    @staticmethod
    def extract_text_from_pdf(pdf_path: str):
        """Extract text from a PDF file."""
        doc = fitz.open(pdf_path)
        text_by_page = []
        for page_num, page in enumerate(doc):
            text_by_page.append((page_num, page.get_text()))
        return text_by_page

    @staticmethod
    def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 100):
        """Split text into chunks of approximately chunk_size words with overlap."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i : i + chunk_size])
            chunks.append(chunk)
        return chunks

def clear_redis_store():
    """Clear the Redis database"""
    print("Clearing existing Redis store...")
    redis_client.flushdb()
    print("Redis store cleared.")

def create_hnsw_index():
    """Create the Redis vector similarity index"""
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass

    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA text TEXT
        embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """
    )
    print("Index created successfully.")
    return "Processing Complete"


def get_embedding(text, embedder):
    """Generate embedding"""
    return embedder.embed_chunks([text])[0] 

@timer
@memory
def search_embeddings(query, embedder, top_k=3):
    """Search the Redis database for similar embeddings."""
    query_embedding = get_embedding(query, embedder)
    
    # Convert embedding to bytes for Redis search
    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

    try:
        q = (
            Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields("file", "page", "text", "vector_distance")
            .dialect(2)
        )

        # Perform the search
        results = redis_client.ft(INDEX_NAME).search(
            q, query_params={"vec": query_vector}
        )

        # Transform results into a structured format
        top_results = [
            {
                "file": result.file,
                "page": result.page,
                "chunk": result.text,
                "similarity": result.vector_distance,
            }
            for result in results.docs
        ][:top_k]

        # Debugging output
        for result in top_results:
            print(f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}")

        return top_results
    except Exception as e:
        print(f"Search error: {e}")
        return []

def main():
    embedders = [
        ("MiniLM", MiniLMEmbedder()),
        ("Instructor", InstructorEmbedder()),
        ("MPNet", MPNetEmbedder())]

    chunk_sizes = [200, 500, 1000]  
    overlaps = [50, 100, 200]  
    # List to store results
    results = []

    # Test all combinations
    for embedder_name, embedder in embedders:
        for i in range(len(chunk_sizes)):
            # Initialize the document processor
            processor = DocumentProcessor(embedder)

            # Clear xisting Redis store
            clear_redis_store()

            create_hnsw_index()

            curr_chunk =  chunk_sizes[i]
            curr_overlap = overlaps[i]

            data_dir = "data"
            (result, memory_used), time_taken = processor.process_pdfs(data_dir, curr_chunk, curr_overlap)
            prompts = [
    "Write a Mongo query based on the movies data set that returns the titles of all movies released between 2010 and 2015 from the suspense genre.",
    "Why is a B+ Tree a better than an AVL tree when indexing a large dataset?",
    "Add 23 to the AVL Tree below.  What imbalance case is created with inserting 23?30/\\ 2535/20",
    "When was Redis originally released?",
    "Insert the key:value pairs sequentially into an initially empty AVL tree: "
]
            for number, prompt in enumerate(prompts):
                (raw_context_result, search_memory), search_time = search_embeddings(prompt, embedder)

                results.append({
                    "DB": "Redis",
                    "Embedder": embedder_name,
                    "Chunk Size": chunk_sizes[i],
                    "Overlap": overlaps[i],
                    "Question Number": number,
                    "Context Retrieval Time": search_time,
                    "Context Retrieval Memory": search_memory,
                })
            print("Processing complete!")
            print(f"Total Time: {time_taken:.2f} ms")
            print(f"Total Memory: {memory_used:.2f} KB")

    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    csv_filename = f'redis_context_performance.csv'
    df.to_csv(csv_filename, index=False)
    print(f"\nPerformance metrics saved to: {csv_filename}")

main()