from InstructorEmbedding import INSTRUCTOR
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer




class InstructorEmbedder:
    def __init__(self):
        """Initialize the DocumentEmbedder with the InstructorXL model"""
        self.model = SentenceTransformer('hkunlp/instructor-xl')
        
    def embed_chunks(self, chunks):
        """Embed a list of text chunks using the INSTRUCTOR model"""
        query_instruction = "Represent the question for retrieving supporting documents:"
        embedding_inputs = [[query_instruction, chunk] for chunk in chunks]
        return self.model.encode(embedding_inputs)
    
    def embed_query(self, query):
        """Embed a query using the instructor model"""
        query_instruction = "Represent the question for retrieving supporting documents:"
        return self.model.encode([[query_instruction, query]])




class MiniLMEmbedder:
    def __init__(self):
        """Initialize the DocumentEmbedder with the MiniLM model"""
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed_chunks(self, chunks):
        """Embed a list of text chunks using the MiniLM model"""
        return self.model.encode(chunks)
    
    def embed_query(self, query, instruction=None):
        """Embed a query using the MiniLM model
        Note: instruction parameter is ignored as MiniLM doesn't use instructions"""
        return self.model.encode([query])




class MPNetEmbedder:
    def __init__(self):
        """Initialize the MPNetEmbedder with the MPNet model"""
        self.model = SentenceTransformer('all-mpnet-base-v2')

    def embed_chunks(self, chunks):
        """Embed a list of text chunks using the MPNet model"""
        return self.model.encode(chunks)
    
    def embed_query(self, query, instruction=None):
        """Embed a query using the MPNet model
        Note: instruction parameter is ignored as MPNet doesn't use instructions"""
        return self.model.encode([query])

