#==============================================================================================
# DO NOT RUN THIS SCRIPT , WE HAVE LOADED 9 VIDEOS TO DB
#==============================================================================================

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import json

# Initialize Pinecone
pc = Pinecone(api_key="pcsk_Ca6aW_SnS7nWDn1Cupww7n7eWLstBW2osqp62V9BcEnUdoqdqMCJ7FPypb5BAs64VzT23")

index_name_keyframe = 'keyframe-index'
index_name_video = 'video-index'

pc.create_index(
    name=index_name_keyframe,
    dimension=384, 
    metric="cosine", 
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)

pc.create_index(
    name=index_name_video,
    dimension=384, 
    metric="cosine", 
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)

# Connect to the Pinecone indexes
keyframe_index = pc.Index("keyframe-index")
video_index = pc.Index("video-index")

# Load text embedding model (e.g., Sentence Transformers)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_vectors_and_save_to_pinecone(keyframe_descriptions_file):
    with open(keyframe_descriptions_file, 'r') as f:
        keyframe_data = json.load(f)

    # Iterate through each video
    for video_id, keyframes in keyframe_data.items():
        print(f"Processing video: {video_id}")

        # Generate video summary by concatenating all keyframe captions
        concatenated_captions = " ".join(
            keyframe["caption"] for keyframe in keyframes.values()
        )
        video_vector = embedding_model.encode(concatenated_captions).tolist()

        # Save video summary vector to the video-index
        video_metadata = {"video_id": video_id, "summary": concatenated_captions}
        video_index.upsert([(video_id, video_vector, video_metadata)])
        print(f"Saved video summary for {video_id}")

        # Process each keyframe
        keyframe_vectors = []
        for keyframe_id, keyframe in keyframes.items():
            keyframe_caption = keyframe["caption"]
            keyframe_timestamp = keyframe["timestamp"]

            # Generate vector for the keyframe caption
            keyframe_vector = embedding_model.encode(keyframe_caption).tolist()

            # Create metadata for the keyframe
            keyframe_metadata = {
                "video_id": video_id,
                "keyframe_id": keyframe_id,
                "caption": keyframe_caption,
                "timestamp": keyframe_timestamp,
            }

            # Save keyframe vector to the keyframe-index
            keyframe_vectors.append((keyframe_id, keyframe_vector, keyframe_metadata))

        # Batch upsert for keyframe vectors
        keyframe_index.upsert(keyframe_vectors)
        print(f"Saved {len(keyframe_vectors)} keyframes for video {video_id}")

keyframe_descriptions_file = "keyframe_descriptions_with_timestamps.json"
generate_vectors_and_save_to_pinecone(keyframe_descriptions_file)