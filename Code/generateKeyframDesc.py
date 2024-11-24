#=============================================================================================================
# extracts keyframes from videos and saves them in keyframes_videokf wilt folder names representing video_ids
#=============================================================================================================


import os
import cv2  # OpenCV for video metadata
import json
from pytubefix import YouTube
from pytubefix.exceptions import VideoUnavailable, VideoPrivate
import videokf as vf

def extract_video_metadata(video_path):
    """Get FPS and duration of the video."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # Total number of frames
    duration = total_frames / fps  # Duration in seconds
    cap.release()
    return fps, duration

# Load training data
train_data_file = "../Dataset/train_videoxum.json"  # Adjust the path if needed
with open(train_data_file, 'r') as file:
    train_data = json.load(file)

# Metadata structure
youtube_data = {
    "url": [],
    "file": [],
    "timestamps": {}
}

# Process each video
for i in range(len(train_data[:10])):
    datapoint = train_data[i]
    video_id = datapoint["video_id"]
    video_file = video_id.split("_")[-1]
    youtube_file = f"https://www.youtube.com/watch?v={video_file}"

    # Ensure directories exist
    file_dir = "../Dataset/video_folder/"
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    keyframe_base_dir = "../Dataset/keyframes_videokf/"
    if not os.path.exists(keyframe_base_dir):
        os.makedirs(keyframe_base_dir)

    file_path = os.path.join(file_dir, f"{video_file}.mp4")
    print(f"Processing video: {youtube_file}")

    try:
        # Download video
        yt = YouTube(youtube_file)
        stream = yt.streams.get_highest_resolution()
        stream.download(output_path=file_dir, filename=video_file)
        youtube_data["url"].append(youtube_file)
        youtube_data["file"].append(video_file)
    except VideoUnavailable:
        print(f"The video {youtube_file} is unavailable.")
        continue
    except VideoPrivate:
        print(f"The video {youtube_file} is private.")
        continue
    except Exception as e:
        print(f"An error occurred with {youtube_file}: {e}")
        continue

    # Extract FPS and duration
    fps, duration = extract_video_metadata(file_path)

    # Prepare keyframe folder
    keyframe_folder = os.path.join(keyframe_base_dir, video_file)
    if not os.path.exists(keyframe_folder):
        os.mkdir(keyframe_folder)

    # Extract keyframes
    vf.extract_keyframes(file_path, method="iframes", output_dir_keyframes=keyframe_folder)

    # Map keyframes to timestamps
    keyframe_files = sorted(os.listdir(keyframe_folder))
    timestamps = []
    for keyframe_file in keyframe_files:
        try:
            frame_number = int(keyframe_file.split('.')[0])  # Extract frame number from file name
            timestamp = frame_number / fps
            timestamps.append({"keyframe": keyframe_file, "timestamp": timestamp})
        except ValueError:
            print(f"Skipping invalid keyframe file: {keyframe_file}")

    youtube_data["timestamps"][video_file] = timestamps

# Save metadata
output_metadata_file = "../Dataset/youtube_metadata_with_timestamps.json"
with open(output_metadata_file, 'w') as outfile:
    json.dump(youtube_data, outfile, indent=4)

print(f"Metadata with timestamps saved to {output_metadata_file}")


#=============================================================================================================
# generates text descriptions from keyframes and saves into json file keyframe_descriptions_with_timestamps
#=============================================================================================================


import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import json

# Initialize BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_descriptions_for_all_videos(keyframes_base_folder, timestamps_file, output_file):
    video_descriptions = {}  # To store descriptions per video

    # Load timestamps
    with open(timestamps_file, 'r') as f:
        timestamps_data = json.load(f)

    # Iterate through each video folder
    for video_id in os.listdir(keyframes_base_folder):
        video_folder = os.path.join(keyframes_base_folder, video_id)
        if not os.path.isdir(video_folder):
            continue

        # Check if video_id exists in timestamps_data["timestamps"]
        if "timestamps" not in timestamps_data or video_id not in timestamps_data["timestamps"]:
            print(f"Warning: {video_id} not found in timestamps data.")
            continue

        keyframe_descriptions = {}
        keyframe_filenames = sorted(os.listdir(video_folder))

        for keyframe_filename in keyframe_filenames:
            keyframe_path = os.path.join(video_folder, keyframe_filename)
            image = Image.open(keyframe_path)

            # Generate caption for the keyframe
            inputs = processor(image, return_tensors="pt")
            outputs = model.generate(**inputs)
            caption = processor.decode(outputs[0], skip_special_tokens=True)

            # Get timestamp for the current keyframe
            timestamp_data = next(
                (item for item in timestamps_data["timestamps"][video_id] if item["keyframe"] == keyframe_filename), None
            )
            timestamp = timestamp_data["timestamp"] if timestamp_data else None

            # Create a keyframe metadata object
            keyframe_metadata = {
                "caption": caption,
                "timestamp": timestamp
            }

            # Use a unique ID format: videoid_keyframefilename
            keyframe_id = f"{video_id}_{keyframe_filename}"
            keyframe_descriptions[keyframe_id] = keyframe_metadata
            print(f"Caption for {keyframe_id}: {caption}, Timestamp: {timestamp}")

        # Add keyframe descriptions to the video-level data
        video_descriptions[video_id] = keyframe_descriptions

    # Save all descriptions to a JSON file
    with open(output_file, 'w') as f:
        json.dump(video_descriptions, f, indent=4)

    print(f"Descriptions with timestamps saved to {output_file}")
    return video_descriptions

# Example usage
keyframes_base_folder = "../Dataset/keyframes_videokf" 
timestamps_file = "../Dataset/youtube_metadata_with_timestamps.json" 
output_file = "keyframe_descriptions_with_timestamps.json"
keyframe_descriptions = generate_descriptions_for_all_videos(keyframes_base_folder, timestamps_file, output_file)
