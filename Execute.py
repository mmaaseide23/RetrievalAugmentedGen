from Embedding import InstructorEmbedder, MiniLMEmbedder
from Processing import extract_text_from_pdf, split_text_into_chunks


import requests

class LLMHandler:
    def __init__(self, model_name="llama2"):
        """Initialize the LLM handler with specified model
        
        Args:
            model_name (str): Name of the model to use ('llama2' or 'mistral')
        """
        self.model_name = model_name
        self.api_base = "http://localhost:11434/api"
        
    def set_model(self, model_name):
        """Change the model being used
        
        Args:
            model_name (str): Name of the model to use ('llama2' or 'mistral')
        """
        self.model_name = model_name
        
    def generate_response(self, prompt, context=None, system_prompt=None):
        """Generate a response from the LLM
        
        Args:
            prompt (str): The user's question
            context (str): Retrieved context from vector search
            system_prompt (str): Optional system prompt to guide the model
        """
        # Construct the full prompt
        full_prompt = ""
        if system_prompt:
            full_prompt += f"{system_prompt}\n\n"
        if context:
            full_prompt += f"Context:\n{context}\n\n"
        full_prompt += f"Question: {prompt}\n\nAnswer:"
        
        # Make request to Ollama API
        response = requests.post(
            f"{self.api_base}/generate",
            json={
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False
            }
        )
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            return f"Error: {response.status_code} - {response.text}"

# Example usage
if __name__ == "__main__":
    # # Initialize with Llama 2
    llm = LLMHandler(model_name="llama2")
    
    # # Test with Llama 2
    # response = llm.generate_response(
    #     prompt="What is machine learning?",
    #     context="Machine learning is a subset of artificial intelligence that focuses on creating systems that can learn from data.",
    #     system_prompt="You are a helpful AI assistant that provides clear and concise answers."
    # )
    # print("\nLlama 2 Response:")
    # print(response)
    
    # Switch to Mistral
    llm.set_model("mistral")
    
    # Test with Mistral
    response = llm.generate_response(
        prompt="What is machine learning?",
        context="Machine learning is a subset of artificial intelligence that focuses on creating systems that can learn from data.",
        system_prompt="You are a helpful AI assistant that provides clear and concise answers."
    )
    print("\nMistral Response:")
    print(response) 