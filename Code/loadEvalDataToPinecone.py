from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import pandas as pd

# Initialize Pinecone
pc = Pinecone(api_key="pcsk_5VNmq8_HG7kdRxs5nYLzBoYrQ7K5yS73B7w7QioX8gdJ5DgWiYoKJxh7f2x75kHWNQ6Uuz")

index_sentence_vectors = "sentence-vectors"

pc.create_index(
        name=index_sentence_vectors,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

sentence_vector_index = pc.Index(index_sentence_vectors)

# Load SentenceTransformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_vectors_and_save_to_pinecone(sentence_pair_file):
    # Read Excel file into a DataFrame
    df = pd.read_excel(sentence_pair_file, sheet_name=0)  # Reads the first sheet by default

    sentence_id = 0
    # Iterate through each sentence pair
    for _, row in df.iterrows():
        sentence_query = row['sentence_A']
        sentence_answer = row['sentence_B']

        # Generate embeddings
        sentence_query_vector = embedding_model.encode(sentence_query).tolist()
        sentence_answer_vector = embedding_model.encode(sentence_answer).tolist()

        # Save query to Pinecone
        query_metadata = {"sentence_id": sentence_id, "sentence": sentence_query}
        sentence_vector_index.upsert([(f"query-{sentence_id}", sentence_query_vector, query_metadata)])

        # Save answer to Pinecone
        answer_metadata = {"sentence_id": sentence_id + 1, "sentence": sentence_answer}
        sentence_vector_index.upsert([(f"answer-{sentence_id}", sentence_answer_vector, answer_metadata)])

        print(f"Saved query and answer pair with IDs query-{sentence_id, sentence_query} and answer-{sentence_id, sentence_answer}")
        sentence_id += 2

# Provide the path to your Excel file
sentence_pair_file = "../Dataset/SentencePairsSheet.xlsx"
generate_vectors_and_save_to_pinecone(sentence_pair_file)