# Retrieval-Augmented Generation for DS4300 Notes
## Welcome to the repo for our a Retrieval-Augmented Generation (RAG) system designed to help students review course material by querying their own class notes using locally run LLMs.


Through testing we have determined what we believe to be the optimal combination of ingestion methods, querying, and LLM model. Here we will give you the instructions to use the repo for this
method we have decided upon. Before using this repo, you will need to make sure that you are using an environment with all of the libraries in our requirements.txt and download an AI model from Ollama (preferaly Mistral). Additionally, we currently have our notes uploaded as the data to be used for our system, if you are using this for a different purpose you will have to remove our data and upload the data you would like to be used for the RAG system. The following instructions are for general use, if you would like to use the method that we found optimal you should use the following things when going through the instructions:

Chunk size: 1000
Overlap size: 200
Embedder: InstructorXL
Database: FAISS
AI Model: Mistral or Mixtral



### Ingestion: 
To ingest the data, you need to start by choosing which database you want to use. Once you decide which one you would like to use, move the data folder, embedder.py and measure.py into that folder. 
After that, navigate to the file with ingestion in the name and run it. You have the ability to alter the embedder, chunk size, and overlap towards the top of each file. Otherwise once you run it your data
will be ingested.

### LLM execution: 
Once you have your data ingested, within the database folder you have been working in, you just need to navigate to the file that has the word execute in it. From there, in the function querying the model, you are able to change the model to any model in Ollama that you have installed. Once you make your choice (or use the default) just run the file and in your terminal there will be a space for you to query the system. 



### Team: Sam Baldwin, Michael Maaseide, Jeff Krapf, Alex Tu
