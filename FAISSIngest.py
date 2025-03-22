import faiss
import numpy as np
import fitz
import os
import pickle
from Embedding import MiniLMEmbedder, MPNetEmbedder, InstructorEmbedder
from measure import timer, memory
import pandas as pd

class FAISS:
    def __init__(self, collection_name="awesome_collection", embedding_function=None, chunk_size=300, overlap=50):
        """Initialize FAISS database with a collection name and optional embedding function"""
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.index = None
        self.chunks = []
        self.metadata = []
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file, returning list of (page_num, text) tuples"""
        doc = fitz.open(pdf_path)
        text_by_page = []
        for page_num, page in enumerate(doc):
            text_by_page.append((page_num, page.get_text()))
        return text_by_page

    def split_text_into_chunks(self, text):
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = " ".join(words[i : i + self.chunk_size])
            chunks.append(chunk)
        return chunks
    
    def _initialize_index(self, dimension):
        """Initialize the FAISS index with the correct dimension"""
        if self.index is None:
            self.index = faiss.IndexFlatL2(dimension)
    
    def get_embedding_faiss(self, text, n_results=5):
        """Search for similar chunks using the query text"""
        if not self.index:
            raise ValueError("No documents have been added to the index yet")
            
        # Get embedding for the query
        embedding = self.embedding_function([text])[0]
        embedding = embedding.reshape(1, -1).astype(np.float32)
        
        # Search the index
        distances, indices = self.index.search(embedding, n_results)
        
        # Format results
        results = {
            "documents": [self.chunks[i] for i in indices[0]],
            "metadatas": [self.metadata[i] for i in indices[0]],
            "distances": distances[0].tolist()
        }
        return results

    def store_embedding_faiss(self, file, page, chunk, embedding):
        """Store a single chunk and its embedding in the FAISS index"""
        # Initialize index if this is the first embedding
        self._initialize_index(embedding.shape[0])
        
        # Add the embedding to the index
        self.index.add(embedding.reshape(1, -1).astype(np.float32))
        
        # Store the chunk and metadata
        self.chunks.append(chunk)
        self.metadata.append({"file": file, "page": page})

    @timer
    @memory
    def process_pdfs(self, data_dir):
        """Process all PDFs in the given directory"""
        for file_name in os.listdir(data_dir):
            if file_name.endswith(".pdf"):
                pdf_path = os.path.join(data_dir, file_name)
                #print(f"Processing {file_name}")
                
                text_by_page = self.extract_text_from_pdf(pdf_path)
                for page_num, text in text_by_page:
                    chunks = self.split_text_into_chunks(text)
                    
                    for chunk in chunks:
                        embedding = self.embedding_function([chunk])[0]
                        self.store_embedding_faiss(
                            file=file_name,
                            page=str(page_num),
                            chunk=chunk,
                            embedding=embedding
                        )
                
                #print(f"Processed {file_name}")
    
    def save(self, directory):
        """Save the FAISS index and associated data to disk"""
        os.makedirs(directory, exist_ok=True)
        
        # Save the FAISS index
        faiss.write_index(self.index, os.path.join(directory, f"{self.collection_name}.index"))
        
        # Save the chunks and metadata
        with open(os.path.join(directory, f"{self.collection_name}.pkl"), "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "metadata": self.metadata
            }, f)
    
    def load(self, directory):
        """Load a saved FAISS index and associated data from disk"""
        # Load the FAISS index
        self.index = faiss.read_index(os.path.join(directory, f"{self.collection_name}.index"))
        
        # Load the chunks and metadata
        with open(os.path.join(directory, f"{self.collection_name}.pkl"), "rb") as f:
            data = pickle.load(f)
            self.chunks = data["chunks"]
            self.metadata = data["metadata"]

def main():
    # Define embedders to test
    embedders = [
        ("MiniLM", MiniLMEmbedder()),
        ("MPNet", MPNetEmbedder()),
        ("Instructor", InstructorEmbedder())
    ]
    
    # Define configurations to test
    configs = [
        {"chunk_size": 200, "overlap": 50},
        {"chunk_size": 500, "overlap": 100},
        {"chunk_size": 1000, "overlap": 200}
    ]
    
    # List to store results
    results = []
    
    # Test all combinations
    for embedder_name, embedder in embedders:
        for config in configs:
            print(f"\n=== Testing Configuration ===")
            print(f"Embedder: {embedder_name}")
            print(f"Chunk size: {config['chunk_size']}")
            print(f"Overlap: {config['overlap']}")
            
            # Initialize FAISS with current configuration
            faiss_db = FAISS(
                collection_name=f"awesome_collection_{embedder_name}_{config['chunk_size']}_{config['overlap']}",
                embedding_function=embedder.model.encode,
                chunk_size=config['chunk_size'],
                overlap=config['overlap']
            )
            
            # Process PDFs
            data_dir = "Data"
            (result, memory_used), time_taken = faiss_db.process_pdfs(data_dir)
            
            # Store results
            results.append({
                'embedder': embedder_name,
                'chunk_size': config['chunk_size'],
                'overlap': config['overlap'],
                'time_ms': time_taken,
                'memory_kb': memory_used
            })
            
            print("\nResults:")
            print(f"Total Time: {time_taken:.2f} ms")
            print(f"Total Memory: {memory_used:.2f} KB")
            
            # Save the database with configuration-specific name
            save_dir = f"faiss_data_{embedder_name}_{config['chunk_size']}_{config['overlap']}"
            faiss_db.save(save_dir)
            print(f"Saved index to: {save_dir}")
            print("=" * 50)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    csv_filename = f'faiss_performance.csv'
    df.to_csv(csv_filename, index=False)
    print(f"\nPerformance metrics saved to: {csv_filename}")
    

if __name__ == '__main__':
    main() 