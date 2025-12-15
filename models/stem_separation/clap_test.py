import torch
from transformers import AutoTokenizer, ClapModel

model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

inputs = tokenizer(["the sound of a cat", "the sound of a dog"], padding=True, return_tensors="pt")
with torch.inference_mode():
    text_features = model.get_text_features(**inputs)
    print(f"input shape: {inputs}")
    print(f"output shape: {text_features.shape}")