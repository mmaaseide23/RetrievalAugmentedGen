import pandas as pd
import numpy as np
import fitz
import os
import redis
import chromadb
from chromadb.utils import embedding_functions


client = chromadb.Client()

# If you supply an embedding function, you must supply it every time you get the collection. 
# Otherwise, it will use the default embedding function which is sentence transformer L6 v2
# I'm curious how we are going to go about that but I think we just need to pass it in as a string parameter
#client = chromadb.PersistentClient(path="/path/to/save/to") client to utilize when saving the function

collection = client.create_collection(name="awesome_collection", )# if different embedding function is used must tell chroma
default_ef = embedding_functions.DefaultEmbeddingFunction()
# Initialize Redis connection
redis_client = redis.Redis(host="localhost", port=6380, db=0)

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"



# Add preprocessing
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

def process_pdfs(data_dir):
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text)
                # print(f"  Chunks: {chunks}")
                for chunk_index, chunk in enumerate(chunks):
                    # embedding = calculate_embedding(chunk)
                    embedding = get_embedding(chunk)
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        # chunk=str(chunk_index),
                        chunk=str(chunk),
                        embedding=embedding,
                    )
            print(f" -----> Processed {file_name}")



def main():
    data = 1
    


if __name__ == '__main__':
    main()