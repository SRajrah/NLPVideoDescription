import faiss
import numpy as np
import pandas as pd
import json
from sentence_transformers import SentenceTransformer

# Initialize the SentenceTransformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_vectors_and_save_to_faiss(sentence_pair_file, faiss_index_file, metadata_file):
    # Read Excel file into a DataFrame
    df = pd.read_excel(sentence_pair_file, sheet_name=0)

    # Prepare embeddings and metadata
    all_embeddings = []
    metadata = []

    for idx, row in df.iterrows():
        sentence_query = row['sentence_A']
        sentence_answer = row['sentence_B']

        # Generate query and answer embeddings
        query_vector = embedding_model.encode(sentence_query).tolist()
        answer_vector = embedding_model.encode(sentence_answer).tolist()

        # Append embeddings
        all_embeddings.append(query_vector)
        metadata.append({"id": f"query-{idx}", "sentence": sentence_query})

        all_embeddings.append(answer_vector)
        metadata.append({"id": f"answer-{idx}", "sentence": sentence_answer})

    # Convert embeddings to numpy array
    all_embeddings = np.array(all_embeddings).astype('float32')

    # Initialize FAISS index
    dimension = 384  # Embedding dimension
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
    index.add(all_embeddings)  # Add embeddings to FAISS
    
    # Save the FAISS index
    faiss.write_index(index, faiss_index_file)

    # Save metadata to a JSON file
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved FAISS index to {faiss_index_file} and metadata to {metadata_file}")

# File paths
sentence_pair_file = "../Dataset/SentencePairsSheet.xlsx"
faiss_index_file = "../Dataset/faiss_index.index"
metadata_file = "../Dataset/faiss_metadata.json"

# Generate vectors and save
generate_vectors_and_save_to_faiss(sentence_pair_file, faiss_index_file, metadata_file)