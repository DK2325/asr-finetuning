import os
import json
import torchaudio

def get_duration(path):
    """Calculates the duration of a WAV file."""
    try:
        metadata = torchaudio.info(path)
        return metadata.num_frames / metadata.sample_rate
    except Exception as e:
        print(f"Error getting duration for {path}: {e}")
        return 0

def create_manifest(audio_dir, transcript_file, output_manifest):
    """Creates a NeMo-compatible manifest file."""
    print(f"Creating manifest from files in '{audio_dir}'...")
    with open(transcript_file, 'r') as tf, open(output_manifest, 'w') as mf:
        for line in tf:
            parts = line.strip().split("|")
            if len(parts) != 2:
                continue
            
            filename, text = parts
            full_path = os.path.abspath(os.path.join(audio_dir, filename))

            if not os.path.exists(full_path):
                print(f"Warning: Missing audio file, skipping: {full_path}")
                continue

            duration = get_duration(full_path)
            entry = {
                "audio_filepath": full_path,
                "duration": round(duration, 2),
                "text": text.lower()
            }
            mf.write(json.dumps(entry) + "\n")
    print(f"✅ Finished creating {output_manifest}")

create_manifest("train", "transcripts/train.txt", "train_manifest.json")
create_manifest("val", "transcripts/val.txt", "val_manifest.json")