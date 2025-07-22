import os
import nemo.collections.asr as nemo_asr
import json

print("Loading models for comparison...")
base_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")
finetuned_model = nemo_asr.models.EncDecCTCModel.restore_from("asr_finetuned.nemo")

val_folder = "val/"
audio_files = [os.path.join(val_folder, f) for f in os.listdir(val_folder) if f.endswith(".wav")]

ground_truths = {}
print("Loading ground truth transcripts from val_manifest.json...")
with open("val_manifest.json", "r") as f:
    for line in f:
        entry = json.loads(line)
        ground_truths[entry['audio_filepath']] = entry['text']

print(f"Transcribing {len(audio_files)} files with the original model...")
base_results = base_model.transcribe(audio_files)

print(f"Transcribing {len(audio_files)} files with the fine-tuned model...")
finetuned_results = finetuned_model.transcribe(audio_files)


print("\n--- Side-by-Side Transcription Comparison ---")
for i, audio_file_path in enumerate(audio_files):
    base_transcript = base_results[i].text if hasattr(base_results[i], 'text') else base_results[i]
    finetuned_transcript = finetuned_results[i].text if hasattr(finetuned_results[i], 'text') else finetuned_results[i]
    ground_truth_text = ground_truths.get(os.path.abspath(audio_file_path), "--- Ground truth not found ---")

    print(f"File:           {os.path.basename(audio_file_path)}")
    print(f"Original Text:  {ground_truth_text}")
    print(f"Before (Base):  {base_transcript}")
    print(f"After (Tuned):  {finetuned_transcript}\n")

