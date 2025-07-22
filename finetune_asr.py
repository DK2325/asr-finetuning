import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

print("Loading pre-trained QuartzNet model...")
asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")

labels = list(" abcdefghijklmnopqrstuvwxyz'")

print("Setting up training data...")
asr_model.setup_training_data(train_data_config={
    "manifest_filepath": "train_manifest.json",
    "batch_size": 4,
    "shuffle": True,
    "labels": labels,
    "sample_rate": 16000
})

print("Setting up validation data...")
asr_model.setup_validation_data(val_data_config={
    "manifest_filepath": "val_manifest.json",
    "batch_size": 4,
    "labels": labels,
    "sample_rate": 16000
})

trainer_cfg = {
    "max_epochs": 1,     
    "accelerator": "cpu",   
    "devices": 1            
}

print("Initializing trainer for CPU...")
trainer = Trainer(**trainer_cfg)
asr_model._trainer = trainer

print("Starting training...")
asr_model.train()
print("✅ Training complete.")

asr_model.save_to("asr_finetuned.nemo")
print("Model saved to asr_finetuned.nemo")