# audio-to-sheet-music
CS 7150 Final Project: Audio to Sheet Music Conversion using Transformer models

<div align="center">
  <h1>Audio to Sheet Music Transcription</h1>
  <p><i>Converting Audio Files</i></p>

  <!-- PyTorch Badge -->
  <a href="https://pytorch.org/" target="_blank" style="text-decoration: none; display: inline-block;">
    <img src="https://img.shields.io/badge/PyTorch-2.6-red?logo=pytorch&style=flat-square" />
  </a>

  <!-- License Badge -->
  <a href="https://opensource.org/licenses/MIT" target="_blank" style="text-decoration: none; display: inline-block;">
    <img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" />
  </a>

  <!-- Dataset Badge -->
  <a href="https://zenodo.org/records/4599666" target="_blank" style="text-decoration: none; display: inline-block;">
    <img src="https://img.shields.io/badge/Dataset-MusDB18-8A2BE2?style=flat-square&logo=gitbook&logoColor=white&labelColor=gray" />
  </a>
</div>

---
This repository tackles audio to sheet music conversion, specifically focusing on stem/track
separation and MIDI conversion
![AMT_Workflow](Audio_Machine_Translation_Flowchart.png)

## Python Environment
It is highly recommended to create a new Python virtual environment to 
run this repo. If using `conda`, the following commands can be used to create the new environment:
```
conda create -n cs_7150_stem_sep python=3.13
```

As PyTorch is the ML framework used for this project, follow the [PyTorch instructions](https://pytorch.org/get-started/previous-versions/) (with CUDA support if desired) to install v2.6.0:
```
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

All remaining dependencies can be installed using pip:
```
pip install -r requirements.txt
```

## Dataset
We are using the MusDB18 dataset. We create our `torch.Dataset` class
for it. It loads in all segments (of length 6s) for all stems for
all songs. For the text prompts, it uses the original stem name
(drums, bass, vocals, other) as well as slight variants
(i.e. for vocal, "vocals", "voice", "singing", "the vocals" are used as
additional prompts)

## Training
To train the model, run the Python scipt `main.py` from the projet root 
directory.
All data, model, and logging configurations are specified in the `config.yaml` file. The full training loop can be found in `src/train.py`.
The YAML configuration file is organized by data,
model, training, and Weights and Biases (wandb) parameters.

The loss functions used are Signal Distortion Ratio (SDR), 
Scale-Invariant SDR (SISDR), a combined loss function using a
linear combination of the SDR and SISDR loss, and lastly a positive 
SDR loss.

For logging purposes during training, we use a Weights & Biases project dashboard.
You will need to create an account using the instructions [here](https://docs.wandb.ai/models/quickstart#python).

## Inference
As a pre-requisite, you must have the model checkpoint file.
To avoid training from scratch, the [latest model file](https://northeastern-my.sharepoint.com/:f:/r/personal/krucinski_j_northeastern_edu/Documents/CS%207150%20Final%20Project/Models/2025_12_01_batch4?csf=1&web=1&e=yPVYLf) 
can be found on the shared OneDrive.

To perform inference, use the following procedure:
1. Create an `inference` folder, either in the dataset directory or in the project root directory
2. Add ONE .stem.mp4 audio file to the `inference` directory from step 1
3. Run the `test_inference.py` script from the root directory, and change paths in the YAML file as desired

The extracted stems for the entire track length will be placed in a subfolder of the specified
`results` directory with the name of the track.

To experiment with zero-shot inference on unseen stems, edit line 15
of `test_inference.py`. The inference script loops through all these
strings and passes them as prompts to the trained model.

Existing results collected by Jacob can be found on the shared [OneDrive](https://northeastern-my.sharepoint.com/:f:/r/personal/krucinski_j_northeastern_edu/Documents/CS%207150%20Final%20Project/Results?csf=1&web=1&e=4Irmd9).