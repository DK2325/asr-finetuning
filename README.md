# Fine-Tuning a QuartzNet ASR Model with NVIDIA NeMo

## Overview

This project provides a comprehensive pipeline for fine-tuning a pre-trained NVIDIA NeMo QuartzNet model for Automatic Speech Recognition (ASR). The included scripts automate the process of data preparation, manifest generation, model training, and inference, allowing for specialization of the model on a custom English-language audio dataset.

## Features

- **End-to-End Workflow**: Scripts to manage the entire fine-tuning pipeline
- **Data Processing**: Automated generation of NeMo-compatible JSON manifest files from raw audio and text
- **Model Training**: Fine-tunes the QuartzNet15x5Base-En model
- **Inference**: Includes a script to test the fine-tuned model on new audio files

## Prerequisites

- Python 3.10.12
- Ubuntu 22.04.5 LTS (or other Unix-like environment such as macOS, WSL on Windows)

## Installation

To set up the project environment, follow these steps:

1. **Create a Python virtual environment**: This isolates the project's dependencies.
   ```bash
   python3 -m venv venv
   ```

2. **Activate the virtual environment**:
   ```bash
   source venv/bin/activate
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Workflow

### Step 1: Data Preparation

Place your custom audio data into two subdirectories within the main project folder:

- `train/`: Contains all .wav files for training the model
- `val/`: Contains all .wav files for validating the model's performance

The audio files should be in WAV format, preferably with a 16kHz sample rate.

### Step 2: Build Transcript Files

This step is crucial for mapping your audio files to their corresponding transcriptions. It reads the original LibriSpeech .trans.txt files to find the correct text for each audio file you've placed in the train and val folders.

Run the build_transcripts.py script:
```bash
python3 build_transcripts.py
```

This will generate two files, `transcripts/train.txt` and `transcripts/val.txt`, which are required for the next step.

### Step 3: Create Data Manifests

With the transcript lists created, you can now generate the final JSON manifest files that NeMo uses to load the data.

Run the create_manifest.py script:
```bash
python3 create_manifest.py
```

This script will produce `train_manifest.json` and `val_manifest.json`.

### Step 4: Model Fine-Tuning

This is the core training step. The finetune_asr.py script will load the pre-trained QuartzNet model and fine-tune it using your custom data manifests.

Execute the script:
```bash
python3 finetune_asr.py
```

This process is computationally intensive and may take a significant amount of time. Upon completion, it will save the trained model to a file named `asr_finetuned.nemo`.

### Step 5: Inference

After training, you can test your new model's performance by running inference on an audio file from your validation set.

Execute the inference script:
```bash
python3 inference_test.py
```

This script loads your `asr_finetuned.nemo` model and prints its transcription of the specified audio file. You can edit the script to point to different audio files for further testing.

## Evaluation

Evaluation aims to quantify the performance difference between the base ASR model (QuartzNet15x5Base-En) and the fine-tuned model. We used a dedicated validation dataset for this assessment.

We observed the following transcription output for the audio file `3857-180923-0006.wav`:

**Original Text**: not like the other preparation alone but also fulfilment the realization of how typical it all was of that generation and that time brings the clearest understanding of the real scope of the civil war

**Before (Base)**: not like the other preparation alone but also fulfilment the realization of how typical it all was of that generation and that time brings the clearest understanding of the real scope of the civil war

**After (Tuned)**: not like the other preparation alone but also fulfilment the realization of how typical it all was of that generation and that time brings the clearest understanding of the real scope of the civil war

### Interpretation

For this specific audio segment, both the base ASR model (before fine-tuning) and the fine-tuned ASR model produced transcriptions that are identical to the original ground truth text. This indicates that for this particular utterance, both models achieved perfect accuracy (0% Word Error Rate). The base model was already highly accurate for this example, and the fine-tuning process successfully maintained this level of precision without introducing any errors or regressions for this specific file.

## File Structure

```
project/
├── finetune_asr.py          # The main script for training the ASR model
├── inference_test.py        # A script to perform inference using the fine-tuned model
├── create_manifest.py       # Generates the final JSON manifest files for NeMo
├── build_transcripts.py     # Prepares the train.txt and val.txt files needed by the manifest creator
├── requirements.txt         # A list of all required Python packages for the project
├── train/                   # Directory for training audio (.wav) files
├── val/                     # Directory for validation audio (.wav) files
└── transcripts/             # Directory containing the generated train.txt and val.txt files
```