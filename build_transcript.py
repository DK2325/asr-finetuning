import os
import random

librispeech_root = "/home/cloud-user/Downloads/ASR/sample_subset_wav" 
audio_data_dirs = ["train", "val"]
print("Building transcript files...")

transcripts_map = {}
for root, _, files in os.walk(librispeech_root):
    for file in files:
        if file.endswith(".trans.txt"):
            with open(os.path.join(root, file), "r") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        utterance_id, transcript = parts
                        transcripts_map[utterance_id] = transcript

os.makedirs("transcripts", exist_ok=True)

for data_dir in audio_data_dirs:
    matched_data = []
    current_audio_dir = os.path.join(os.getcwd(), data_dir)
    for filename in os.listdir(current_audio_dir):
        if filename.endswith(".wav"):
            utterance_id = os.path.splitext(filename)[0]
            if utterance_id in transcripts_map:
                transcript = transcripts_map[utterance_id]
                matched_data.append(f"{filename}|{transcript.lower()}")
    
    output_path = f"transcripts/{data_dir}.txt"
    with open(output_path, "w") as f:
        f.write("\n".join(matched_data))
    
    print(f"✅ Created {output_path} with {len(matched_data)} records.")