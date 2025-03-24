import pandas as pd
import numpy as np
import fitz
import os
import redis
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from measure import timer, memory

class Chroma:
    def __init__(self, collection_name="awesome_collection", embedding_function=None):
        self.collection_name = collection_name
        
        self.client = chromadb.Client()
        
        if embedding_function is None:
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        else:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_function)
        
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function
        )

    def extract_text_from_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        text_by_page = []
        for page_num, page in enumerate(doc):
            text_by_page.append((page_num, page.get_text()))
        return text_by_page

    def split_text_into_chunks(self, text, chunk_size=300, overlap=50):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i : i + chunk_size])
            chunks.append(chunk)
        return chunks

    def get_embedding_chroma(self, text, n_results=10):
        embedding = self.embedding_function([text])[0]
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results
        )

    def store_embedding_chroma(self, file, page, chunk, embedding):
        self.collection.add(
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{"file": file, "page": page}],
            ids=[f"{file}:{page}:{hash(chunk)}"]
        )
    @timer
    @memory
    def process_pdfs(self, data_dir, chunk_size, overlap):
        for file_name in os.listdir(data_dir):
            if file_name.endswith(".pdf"):
                pdf_path = os.path.join(data_dir, file_name)
                print(f"Processing {file_name}")
                
                text_by_page = self.extract_text_from_pdf(pdf_path)
                for page_num, text in text_by_page:
                    chunks = self.split_text_into_chunks(text, chunk_size, overlap)
                    
                    for chunk in chunks:
                        embedding = self.embedding_function([chunk])[0]
                        self.store_embedding_chroma(
                            file=file_name,
                            page=str(page_num),
                            chunk=chunk,
                            embedding=embedding
                        )
                
                print(f"Processed {file_name}")

def main():

    chroma = Chroma(collection_name="awesome_collection")
    
    data_dir = "Data" 
    chroma.process_pdfs(data_dir)

def main():
    # Define embedders to test
    embedders = [
         ("Instructor", "hkunlp/instructor-xl"),
        ("MiniLM", None),
        ("MPNet", "all-mpnet-base-v2")
       
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
            faiss_db = Chroma(
                collection_name=f"awesome_collection_{embedder_name}_{config['chunk_size']}_{config['overlap']}",
                embedding_function=embedder
            )
            
            # Process PDFs
            data_dir = "Data"
            (result, memory_used), time_taken = faiss_db.process_pdfs(data_dir, config["chunk_size"], config['overlap'])
            
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
            
  
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    csv_filename = f'chroma_performance.csv'
    df.to_csv(csv_filename, index=False)
    print(f"\nPerformance metrics saved to: {csv_filename}")
    
if __name__ == '__main__':
    main()