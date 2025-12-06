"""
Gradio Demo for AudioTextHTDemucs - Text-Conditioned Stem Separation

Upload an audio file, enter a text prompt (e.g., "drums", "extract bass", "vocals"),
and the model will separate that stem from the mixture.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gradio as gr
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from demucs import pretrained
from transformers import ClapModel, AutoTokenizer

from src.models.stem_separation.ATHTDemucs_v2 import AudioTextHTDemucs
from utils import load_config, plot_spectrogram

# ============================================================================
# Configuration
# ============================================================================

cfg = load_config("config.yaml")
CHECKPOINT_PATH     = cfg["training"]["resume_from"]  # Change as needed
SAMPLE_RATE         = cfg["data"]["sample_rate"]
SEGMENT_SECONDS     = cfg["data"]["segment_seconds"]
OVERLAP             = cfg["data"]["overlap"]

# Auto-detect device
# if torch.cuda.is_available():
#     DEVICE = "cuda"
# elif torch.backends.mps.is_available():
#     DEVICE = "mps"
# else:
#     DEVICE = "cpu"
DEVICE = "cpu"


# ============================================================================
# Model Loading
# ============================================================================

print(f"Loading model on device: {DEVICE}")
print("Loading HTDemucs...")
htdemucs = pretrained.get_model('htdemucs').models[0]

print("Loading CLAP...")
clap = ClapModel.from_pretrained("laion/clap-htsat-unfused")
tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

print("Building AudioTextHTDemucs...")
model = AudioTextHTDemucs(htdemucs, clap, tokenizer)

print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

model = model.to(DEVICE)
model.eval()
print("Model ready!")


# ============================================================================
# Helper Functions
# ============================================================================

def create_spectrogram(audio, sr=SAMPLE_RATE, title="Spectrogram"):
    """Create a spectrogram visualization."""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Convert to mono for visualization if stereo
    if audio.shape[0] == 2:
        audio_mono = audio.mean(dim=0)
    else:
        audio_mono = audio.squeeze()
    
    # Compute spectrogram
    n_fft = 2048
    hop_length = 512
    spec = torch.stft(
        audio_mono,
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True
    )
    spec_mag = torch.abs(spec)
    spec_db = 20 * torch.log10(spec_mag + 1e-8)
    
    # Plot
    im = ax.imshow(
        spec_db.cpu().numpy(),
        aspect='auto',
        origin='lower',
        cmap='viridis',
        interpolation='nearest'
    )
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('Frequency (bins)')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    
    return fig


def load_audio(audio_path, target_sr=SAMPLE_RATE):
    """Load audio file and resample if necessary."""
    waveform, sr = torchaudio.load(audio_path)
    
    # Resample if necessary
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    # Convert to stereo if mono
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    
    return waveform, target_sr


def chunked_inference(mixture, prompt):
    """Run chunked inference for a single stem."""
    C, T = mixture.shape
    chunk_len = int(SAMPLE_RATE * SEGMENT_SECONDS)
    overlap_frames = int(OVERLAP * SAMPLE_RATE)
    
    output = torch.zeros(C, T, device=DEVICE)
    weight = torch.zeros(T, device=DEVICE)
    
    start = 0
    while start < T:
        end = min(start + chunk_len, T)
        chunk = mixture[:, start:end].unsqueeze(0).to(DEVICE)  # (1, C, chunk_len)
        
        # Pad if needed
        if chunk.shape[-1] < chunk_len:
            pad_amount = chunk_len - chunk.shape[-1]
            chunk = F.pad(chunk, (0, pad_amount))
        
        with torch.no_grad():
            out = model(chunk, [prompt])  # (1, C, chunk_len)
        
        # Ensure output is on the correct device
        out = out.to(DEVICE).squeeze(0)  # (C, chunk_len)
        
        # Trim padding if we added any
        actual_len = end - start
        out = out[:, :actual_len]
        
        # Create fade weights for overlap-add
        fade_len = min(overlap_frames, actual_len // 2)
        chunk_weight = torch.ones(actual_len, device=DEVICE)
        if start > 0 and fade_len > 0:
            # Fade in
            chunk_weight[:fade_len] = torch.linspace(0, 1, fade_len, device=DEVICE)
        if end < T and fade_len > 0:
            # Fade out
            chunk_weight[-fade_len:] = torch.linspace(1, 0, fade_len, device=DEVICE)
        
        output[:, start:end] += out * chunk_weight
        weight[start:end] += chunk_weight
        
        # Move to next chunk with overlap
        start += chunk_len - overlap_frames
    
    # Normalize by weights
    weight = weight.clamp(min=1e-8)
    output = output / weight
    
    return output

def download_youtube_audio(yt_link):
    """Download audio from a YouTube link using yt-dlp."""
    try:
        import yt_dlp
        os.remove("temp/yt_audio.webm") if os.path.exists("temp/yt_audio.webm") else None
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'quiet': True,
            'outtmpl': 'temp/yt_audio.webm',
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([yt_link])
        
        mixture, sr = load_audio("temp/yt_audio.webm", target_sr=SAMPLE_RATE)
        return (sr, mixture.T.numpy())
    except Exception as e:
        return f"Error downloading audio from YouTube: {str(e)}"


# ============================================================================
# Gradio Interface Functions
# ============================================================================

def process_audio(audio_file, yt_link, text_prompt):
    """Main processing function for the Gradio interface."""
    if audio_file is None and (yt_link is None or yt_link.strip() == ""):
        return None, None, None, None, "Please upload an audio file."
    
    if not text_prompt or text_prompt.strip() == "":
        return None, None, None, None, "Please enter a text prompt."
    
    if yt_link and yt_link.strip() != "":
        try:
            download_youtube_audio(yt_link)
        except Exception as e:
            return None, None, None, None, str(e)
    
    try:
        # Load audio
        mixture, sr = load_audio(audio_file if audio_file else "temp/yt_audio.webm", target_sr=SAMPLE_RATE)
        print(f"Loaded audio: {mixture.shape}, sr={sr}")
        
        # Create input spectrogram
        #input_spec_fig = create_spectrogram(mixture, sr, title="Input Mixture Spectrogram")
        input_spec_fig = plot_spectrogram(mixture, sr, title="Input Mixture Spectrogram")
        
        # Run separation
        print(f"Running separation with prompt: '{text_prompt}'")
        separated = chunked_inference(mixture.to(DEVICE), text_prompt.strip())
        separated = separated.cpu()
        
        # Debug: Check if output is non-zero
        print(f"Separated audio shape: {separated.shape}")
        print(f"Separated audio range: [{separated.min():.4f}, {separated.max():.4f}]")
        print(f"Separated audio mean abs: {separated.abs().mean():.4f}")
        
        # Create output spectrogram
        output_spec_fig = create_spectrogram(separated, sr, title=f"Separated: {text_prompt}")
        
        # Convert to audio format for Gradio
        # Gradio Audio expects tuple: (sample_rate, numpy_array)
        # numpy_array shape should be (samples, channels) for stereo
        input_audio = (sr, mixture.T.numpy())  # (sr, (T, 2))
        output_audio = (sr, separated.T.numpy())  # (sr, (T, 2))
        
        status = f"✓ Successfully separated '{text_prompt}' from the mixture!"
        
        return input_audio, output_audio, input_spec_fig, output_spec_fig, status
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, None, None, None, error_msg


# ============================================================================
# Gradio Interface
# ============================================================================

def create_demo():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="AudioTextHTDemucs Demo") as demo:
        gr.Markdown(
            """
            # 🎵 AudioTextHTDemucs - Text-Conditioned Stem Separation
            
            Upload an audio file and enter a text prompt to separate specific stems from the mixture.
            
            **Example prompts:**
            - `drums` - Extract drum sounds
            - `bass` - Extract bass guitar
            - `vocals` - Extract singing voice
            - `other` - Extract other instruments
            - Or any natural language description like "extract the guitar" or "piano sound"
            """
        )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input")
                audio_input = gr.Audio(
                    label="Upload Audio File",
                    type="filepath",
                    sources=["upload"]
                )
                yt_link_input = gr.Textbox(
                    label="YouTube Video URL (optional)",
                    placeholder="Provide a YouTube link to fetch audio",
                    lines=1
                )
                text_input = gr.Textbox(
                    label="Text Prompt",
                    placeholder="Enter what you want to extract (e.g., 'drums', 'vocals', 'bass')",
                    lines=1
                )
                gr.Examples(
                    examples=[
                        ["drums"],
                        ["bass"],
                        ["vocals"],
                        ["other"],
                        ["extract the drums"],
                        ["guitar sound"],
                    ],
                    inputs=text_input,
                    label="Click to use example prompts"
                )
                
                with gr.Row():
                    clear_btn = gr.Button("Clear", variant="secondary")
                    submit_btn = gr.Button("Separate Audio", variant="primary")
                
                status_output = gr.Textbox(label="Status", interactive=False)
                yt_link_input.change(download_youtube_audio, inputs=[yt_link_input], outputs=[audio_input])
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input Mixture")
                input_audio_player = gr.Audio(
                    label="Input Audio (Original Mix)",
                    type="numpy",
                    interactive=False
                )
                input_spec_plot = gr.Plot(label="Input Spectrogram")
            
            with gr.Column():
                gr.Markdown("### Separated Output")
                output_audio_player = gr.Audio(
                    label="Separated Audio",
                    type="numpy",
                    interactive=False
                )
                output_spec_plot = gr.Plot(label="Output Spectrogram")
        
        # Button actions
        submit_btn.click(
            fn=process_audio,
            inputs=[audio_input, yt_link_input, text_input],
            outputs=[
                input_audio_player,
                output_audio_player,
                input_spec_plot,
                output_spec_plot,
                status_output
            ]
        )
        
        def clear_all():
            return None, "", None, None, None, None, ""
        
        clear_btn.click(
            fn=clear_all,
            outputs=[
                audio_input,
                text_input,
                input_audio_player,
                output_audio_player,
                input_spec_plot,
                output_spec_plot,
                status_output
            ]
        )
        
        gr.Markdown(
            """
            ---
            ### Notes
            - The model works best with music audio sampled at 44.1kHz
            - Processing time depends on audio length (segments processed in 6-second chunks)
            - The model was trained on stems: drums, bass, vocals, and other instruments
            - You can use natural language descriptions thanks to CLAP text embeddings
            """
        )
    
    return demo


# ============================================================================
# Launch
# ============================================================================

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        share=False,  # Set to True to create a public link
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860
    )