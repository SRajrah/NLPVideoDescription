import json
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import pandas as pd

pc = Pinecone(api_key="pcsk_5VNmq8_HG7kdRxs5nYLzBoYrQ7K5yS73B7w7QioX8gdJ5DgWiYoKJxh7f2x75kHWNQ6Uuz")
index_sentence_vectors = "sentence-vectors"
sentence_vector_index = pc.Index(index_sentence_vectors)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def perform_semantic_evaluation(sentence_pair_file, json_output_file):
    results = []

    # Read Excel file into a DataFrame
    df = pd.read_excel(sentence_pair_file, sheet_name=0)  # Reads the first sheet by default

    for _, row in df.iterrows():
        sentence_query = row['sentence_A']
        sentence_answer = row['sentence_B']

        # Generate embedding for the query sentence
        query_vector = embedding_model.encode(sentence_query).tolist()

        # Perform search on Pinecone
        top_k = 5
        search_results = sentence_vector_index.query(vector=query_vector, top_k=top_k, include_metadata=True)

        # Extract matching sentences
        matching_sentences = []
        for rank, match in enumerate(search_results['matches']):
            matching_sentences.append({
                "sentenceId": match['id'],
                "sentence": match['metadata']['sentence'],
                "rank": rank + 1,
                "sim_score": match['score']
            })

        # Append result to JSON structure
        results.append({
            "querySentence": {
                "sentenceId": f"query-{sentence_query}",
                "sentence": sentence_query
            },
            "matchingSentences": matching_sentences
        })

    # Save results to JSON
    with open(json_output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {json_output_file}")

# Perform semantic evaluation and save results
json_output_file = "../Dataset/semantic_results.json"
sentence_pair_file = "../Dataset/SentencePairsSheet.xlsx"
perform_semantic_evaluation(sentence_pair_file, json_output_file)