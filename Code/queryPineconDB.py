from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

pc = Pinecone(api_key="pcsk_Ca6aW_SnS7nWDn1Cupww7n7eWLstBW2osqp62V9BcEnUdoqdqMCJ7FPypb5BAs64VzT23")
# Connect to the Pinecone index
video_index = pc.Index("video-index")

# Load the same embedding model used for data insertion
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def query_pinecone(query, top_k=5):
    # Convert the query to a vector
    query_vector = embedding_model.encode(query).tolist()

    # Perform a similarity search on Pinecone
    response = video_index.query(
        vector=query_vector,
        top_k=top_k,  # Number of top results to retrieve
        include_metadata=True  # Include metadata in the response
    )

    # Extract and print video IDs from the response
    print("Top video matches:")
    for match in response["matches"]:
        video_id = match["metadata"]["video_id"]
        print(f"Video ID: {video_id}, Score: {match['score']}")

# Example query
# query = "a woman dancing on a floor"
query = "a girl getting her ears pierced"
query_pinecone(query)