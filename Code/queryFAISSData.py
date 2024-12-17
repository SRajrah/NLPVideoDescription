import faiss
import numpy as np
import pandas as pd
import json
from sentence_transformers import SentenceTransformer

# Load SentenceTransformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def query_faiss_and_save_results(sentence_pair_file, faiss_index_file, metadata_file, json_output_file):
    # Load FAISS index and metadata
    index = faiss.read_index(faiss_index_file)
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Read Excel file into DataFrame
    df = pd.read_excel(sentence_pair_file, sheet_name=0)

    # Perform queries
    results = []
    for idx, row in df.iterrows():
        sentence_query = row['sentence_A']

        # Generate query embedding
        query_vector = embedding_model.encode(sentence_query).astype('float32')
        query_vector = np.expand_dims(query_vector, axis=0)

        # Search FAISS for top 5 results
        D, I = index.search(query_vector, k=5)

        # Extract matching sentences
        matching_sentences = []
        for rank, idx in enumerate(I[0]):
            if idx >= 0:  # Valid index
                matching_sentences.append({
                    "sentenceId": metadata[idx]["id"],
                    "sentence": metadata[idx]["sentence"],
                    "rank": rank + 1,
                    "sim_score": float(1 / (1 + D[0][rank]))  # Convert to Python float
                })

        # Append result
        results.append({
            "querySentence": {
                "sentenceId": f"query-{idx}",
                "sentence": sentence_query
            },
            "matchingSentences": matching_sentences
        })

    # Save results to JSON
    with open(json_output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {json_output_file}")
# File paths
json_output_file = "../Dataset/faiss_results.json"
# File paths
sentence_pair_file = "../Dataset/SentencePairsSheet.xlsx"
faiss_index_file = "../Dataset/faiss_index.index"
metadata_file = "../Dataset/faiss_metadata.json"
# Query FAISS and save results
query_faiss_and_save_results(sentence_pair_file, faiss_index_file, metadata_file, json_output_file)