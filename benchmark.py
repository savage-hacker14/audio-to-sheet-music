"""
Inference and Evaluation Script for Stem Separation Models

Evaluates the following models on the MusDB18 test set:
1. AudioTextHTDemucs (our model)
2. HTDemucs (baseline)
3. CLAPSep / AudioSep (baseline) - placeholder for text-conditioned baseline

Computes SDR for: drums, bass, vocals, other, and average
"""

import os
# Silence tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import contextlib
import io
import json
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import numpy as np
import soundfile as sf
from tqdm import tqdm
from torchaudio.transforms import Fade
from huggingface_hub import hf_hub_download

# Local imports
from src.models.stem_separation.ATHTDemucs_v2 import AudioTextHTDemucs
from src.loss import sdr_loss, sisdr_loss, new_sdr_metric
from src.dataloader import MusDBStemDataset, STEM_PROMPTS
from utils import (
    plot_separation_spectrograms,
    plot_all_stems_spectrograms,
    log_spectrogram_to_wandb,
    log_all_stems_to_wandb,
)

# Demucs imports
from demucs import pretrained
from demucs.apply import apply_model
from transformers import ClapModel, AutoTokenizer


# ============================================================================
# Configuration
# ============================================================================

STEMS = ["drums", "bass", "other", "vocals"]
SAMPLE_RATE = 44100


@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    test_dir: str = "musdb18/test"
    checkpoint_path: str = "/Users/surya/Downloads/2025_12_01_batch4/best_model.pt"
    output_dir: str = "results"
    segment_seconds: float = 6.0
    overlap: float = 0.1
    device: str = "mps"  # Apple Silicon (M1/M2/M3)
    save_audio: bool = False
    plot_spectrograms: bool = False  # Whether to generate and log spectrograms
    use_wandb: bool = False  # Whether to log spectrograms to W&B
    models_to_eval: List[str] = field(default_factory=lambda: ["ours", "htdemucs", "clapsep"])


# ============================================================================
# Base Model Interface
# ============================================================================

class SeparationModel(ABC):
    """Abstract base class for separation models."""
    
    @abstractmethod
    def separate(self, mixture: torch.Tensor, stem_name: str) -> torch.Tensor:
        """
        Separate a specific stem from the mixture.
        
        Args:
            mixture: (C, T) stereo mixture waveform
            stem_name: One of "drums", "bass", "other", "vocals"
            
        Returns:
            (C, T) separated stem waveform
        """
        pass
    
    @abstractmethod
    def separate_all(self, mixture: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Separate all stems from the mixture.
        
        Args:
            mixture: (C, T) stereo mixture waveform
            
        Returns:
            Dict mapping stem names to (C, T) waveforms
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model name for logging."""
        pass


# ============================================================================
# Our Model: AudioTextHTDemucs
# ============================================================================

class OurModel(SeparationModel):
    """Wrapper for our AudioTextHTDemucs model."""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device
        self.segment_seconds = 6.0
        self.overlap = 1.5  # 1.5 second overlap (~25%) to match HTDemucs
        
        # Load pretrained components
        print("Loading pretrained HTDemucs for our model...")
        htdemucs = pretrained.get_model('htdemucs').models[0]
        
        print("Loading CLAP model...")
        clap = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")
        
        # Build model
        print("Building AudioTextHTDemucs model...")
        self.model = AudioTextHTDemucs(htdemucs, clap, tokenizer)
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
        
        self.model = self.model.to(device)
        self.model.eval()
    
    @property
    def name(self) -> str:
        return "AudioTextHTDemucs (Ours)"
    
    def _chunked_inference(self, mixture: torch.Tensor, prompt: str) -> torch.Tensor:
        """Run chunked inference for a single stem."""
        C, T = mixture.shape
        chunk_len = int(SAMPLE_RATE * self.segment_seconds)
        overlap_frames = int(self.overlap * SAMPLE_RATE)
        
        output = torch.zeros(C, T, device=self.device)
        weight = torch.zeros(T, device=self.device)
        
        start = 0
        while start < T:
            end = min(start + chunk_len, T)
            chunk = mixture[:, start:end].unsqueeze(0).to(self.device)  # (1, C, chunk_len)
            
            # Pad if needed
            if chunk.shape[-1] < chunk_len:
                pad_amount = chunk_len - chunk.shape[-1]
                chunk = F.pad(chunk, (0, pad_amount))
            
            with torch.no_grad():
                out = self.model(chunk, [prompt])  # (1, C, chunk_len)
            
            # Ensure output is on the correct device
            out = out.to(self.device).squeeze(0)  # (C, chunk_len)
            
            # Trim padding if we added any
            actual_len = end - start
            out = out[:, :actual_len]
            
            # Create fade weights for overlap-add
            fade_len = min(overlap_frames, actual_len // 2)
            chunk_weight = torch.ones(actual_len, device=self.device)
            if start > 0 and fade_len > 0:
                # Fade in
                chunk_weight[:fade_len] = torch.linspace(0, 1, fade_len, device=self.device)
            if end < T and fade_len > 0:
                # Fade out
                chunk_weight[-fade_len:] = torch.linspace(1, 0, fade_len, device=self.device)
            
            output[:, start:end] += out * chunk_weight
            weight[start:end] += chunk_weight
            
            # Move to next chunk with overlap
            start += chunk_len - overlap_frames
        
        # Normalize by weights
        weight = weight.clamp(min=1e-8)
        output = output / weight
        
        return output
    
    def separate(self, mixture: torch.Tensor, stem_name: str) -> torch.Tensor:
        mixture = mixture.to(self.device)
        return self._chunked_inference(mixture, stem_name)
    
    def separate_all(self, mixture: torch.Tensor) -> Dict[str, torch.Tensor]:
        mixture = mixture.to(self.device)
        results = {}
        for stem in STEMS:
            results[stem] = self._chunked_inference(mixture, stem)
        return results


# ============================================================================
# Baseline: HTDemucs
# ============================================================================

class HTDemucsModel(SeparationModel):
    """Wrapper for HTDemucs baseline model."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        print("Loading HTDemucs baseline model...")
        self.model = pretrained.get_model('htdemucs').models[0]
        self.model = self.model.to(device)
        self.model.eval()
        
        # HTDemucs outputs sources in order: drums, bass, other, vocals
        self.source_names = self.model.sources  # ['drums', 'bass', 'other', 'vocals']
    
    @property
    def name(self) -> str:
        return "HTDemucs (Baseline)"
    
    def separate_all(self, mixture: torch.Tensor) -> Dict[str, torch.Tensor]:
        mixture = mixture.to(self.device)
        
        # HTDemucs expects (batch, channels, time)
        mix_batch = mixture.unsqueeze(0)  # (1, C, T)
        
        # Normalize
        ref = mixture.mean(0)
        mix_norm = (mix_batch - ref.mean()) / (ref.std() + 1e-8)
        
        with torch.no_grad():
            # apply_model handles chunking internally
            sources = apply_model(self.model, mix_norm, split=True, overlap=0.25, progress=False)
        
        # Denormalize
        sources = sources * ref.std() + ref.mean()
        sources = sources.squeeze(0)  # (num_sources, C, T)
        
        # Map to dict
        results = {}
        for i, stem in enumerate(self.source_names):
            results[stem] = sources[i]
        
        return results
    
    def separate(self, mixture: torch.Tensor, stem_name: str) -> torch.Tensor:
        all_stems = self.separate_all(mixture)
        return all_stems[stem_name]


# ============================================================================
# Baseline: CLAPSep
# ============================================================================

# CLAPSep operates at 32kHz sample rate
CLAPSEP_SAMPLE_RATE = 32000


class CLAPSepModel(SeparationModel):
    """
    Wrapper for CLAPSep baseline model.
    
    Paper: "CLAPSep: Leveraging Contrastive Pre-trained Models for 
           Multi-Modal Query-Conditioned Target Sound Extraction" (Ma et al., 2024)
    
    GitHub: https://github.com/Aisaka0v0/CLAPSep
    HuggingFace: https://huggingface.co/spaces/AisakaMikoto/CLAPSep
    """
    
    def __init__(self, device: str = "cuda", cache_dir: str = "clapsep_model"):
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.model = None
        self._available = False
        
        try:
            self._setup_clapsep()
        except Exception as e:
            print(f"WARNING: Failed to load CLAPSep: {e}")
            print("CLAPSep evaluation will be skipped.")
            self._available = False
    
    def _download_from_huggingface(self):
        """Download CLAPSep model files from HuggingFace."""
        repo_id = "AisakaMikoto/CLAPSep"
        
        # Files we need
        files_to_download = [
            "model/CLAPSep.py",
            "model/CLAPSep_decoder.py",
            "model/best_model.ckpt",
            "model/music_audioset_epoch_15_esc_90.14.pt",
        ]
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print("Downloading CLAPSep model from HuggingFace...")
        for file_path in files_to_download:
            filename = Path(file_path).name
            local_path = self.cache_dir / filename
            
            if not local_path.exists():
                print(f"  Downloading {filename}...")
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_path,
                    repo_type="space",
                    local_dir=self.cache_dir,
                    local_dir_use_symlinks=False,
                )
                # Move from subdirectory to cache_dir root
                src = self.cache_dir / file_path
                if src.exists() and src != local_path:
                    shutil.move(str(src), str(local_path))
            else:
                print(f"  {filename} already cached.")
        
        print("CLAPSep model files downloaded successfully.")
    
    def _fix_relative_imports(self):
        """Fix relative imports in downloaded CLAPSep files."""
        clapsep_file = self.cache_dir / "CLAPSep.py"
        if clapsep_file.exists():
            content = clapsep_file.read_text()
            # Replace relative imports with absolute imports
            if "from .CLAPSep_decoder" in content:
                content = content.replace("from .CLAPSep_decoder", "from CLAPSep_decoder")
                clapsep_file.write_text(content)
                print("  Fixed relative imports in CLAPSep.py")
    
    def _setup_clapsep(self):
        """Set up CLAPSep model."""
        # Download model files if needed
        self._download_from_huggingface()
        
        # Fix relative imports in downloaded files
        self._fix_relative_imports()
        
        # Add cache dir to path so we can import CLAPSep modules
        if str(self.cache_dir) not in sys.path:
            sys.path.insert(0, str(self.cache_dir))
        
        # Import CLAPSep model
        from CLAPSep import CLAPSep
        
        print("Loading CLAPSep model...")
        
        # Model config from CLAPSep repo
        model_config = {
            "lan_embed_dim": 1024,
            "depths": [1, 1, 1, 1],
            "embed_dim": 128,
            "encoder_embed_dim": 128,
            "phase": False,
            "spec_factor": 8,
            "d_attn": 640,
            "n_masker_layer": 3,
            "conv": False,
        }
        
        # Initialize model (suppress verbose CLAP loading output)
        with contextlib.redirect_stdout(io.StringIO()):
            self.model = CLAPSep(
                model_config=model_config,
                CLAP_path=str(self.cache_dir / "music_audioset_epoch_15_esc_90.14.pt"),
            )
        
        # Load pretrained weights
        checkpoint_path = self.cache_dir / "best_model.ckpt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Handle different checkpoint formats
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self._available = True
        print("CLAPSep model loaded successfully.")
    
    @property
    def name(self) -> str:
        return "CLAPSep (Baseline)"
    
    def is_available(self) -> bool:
        return self._available
    
    def _resample(self, audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
        """Resample audio to target sample rate."""
        if orig_sr == target_sr:
            return audio
        
        resampler = T.Resample(orig_sr, target_sr).to(audio.device)
        return resampler(audio)
    
    def _get_text_query(self, stem_name: str) -> str:
        """Get descriptive text query for a stem (CLAPSep trained on AudioCaps descriptions)."""
        # Use natural language descriptions similar to AudioCaps training data
        prompts = {
            "drums": "drums and percussion",
            "bass": "bass guitar and bass sounds",
            "vocals": "singing voice and vocals",
            "other": "other musical instruments",
        }
        return prompts.get(stem_name, stem_name)
    
    def separate(self, mixture: torch.Tensor, stem_name: str) -> torch.Tensor:
        """
        Separate a stem using text query with chunked processing.
        
        Args:
            mixture: (C, T) stereo mixture at 44.1kHz
            stem_name: One of "drums", "bass", "other", "vocals"
            
        Returns:
            (C, T) separated stem at 44.1kHz
        """
        if not self._available:
            raise RuntimeError("CLAPSep model not available")
        
        # CLAPSep works on mono audio at 32kHz
        # Convert stereo to mono
        if mixture.dim() == 2 and mixture.shape[0] == 2:
            mono = mixture.mean(dim=0)  # (T,)
        else:
            mono = mixture.squeeze()  # (T,)
        
        original_length = mono.shape[-1]
        
        # Resample from 44.1kHz to 32kHz
        mono_32k = self._resample(mono.unsqueeze(0), SAMPLE_RATE, CLAPSEP_SAMPLE_RATE).squeeze(0)
        
        # CLAPSep parameters - process in 10 second chunks with overlap
        chunk_seconds = 10.0
        overlap_seconds = 1.0
        chunk_samples = int(chunk_seconds * CLAPSEP_SAMPLE_RATE)
        overlap_samples = int(overlap_seconds * CLAPSEP_SAMPLE_RATE)
        
        total_samples = mono_32k.shape[-1]
        
        # Text query for the stem (use descriptive prompt)
        pos_prompt = self._get_text_query(stem_name)
        neg_prompt = ""  # Empty negative prompt
        
        with torch.no_grad():
            # Get text embeddings from CLAP (only once)
            embed_pos = self.model.clap_model.get_text_embedding([pos_prompt])
            embed_neg = self.model.clap_model.get_text_embedding([neg_prompt])
            embed_pos = torch.from_numpy(embed_pos).to(self.device)
            embed_neg = torch.from_numpy(embed_neg).to(self.device)
            
            # Process in chunks
            output_32k = torch.zeros(total_samples, device=self.device)
            weight = torch.zeros(total_samples, device=self.device)
            
            start = 0
            while start < total_samples:
                end = min(start + chunk_samples, total_samples)
                chunk = mono_32k[start:end]
                
                # Pad if chunk is too short
                if chunk.shape[-1] < chunk_samples:
                    chunk = F.pad(chunk, (0, chunk_samples - chunk.shape[-1]))
                
                # CLAPSep expects (1, T) input
                chunk = chunk.unsqueeze(0).to(self.device)
                
                # CLAPSep inference
                out = self.model.inference_from_data(chunk, embed_pos, embed_neg)
                out = out.squeeze(0)  # (T,)
                
                # Trim to actual length
                actual_len = end - start
                out = out[:actual_len]
                
                # Create fade weights for overlap-add
                fade_len = min(overlap_samples, actual_len // 2)
                chunk_weight = torch.ones(actual_len, device=self.device)
                if start > 0 and fade_len > 0:
                    chunk_weight[:fade_len] = torch.linspace(0, 1, fade_len, device=self.device)
                if end < total_samples and fade_len > 0:
                    chunk_weight[-fade_len:] = torch.linspace(1, 0, fade_len, device=self.device)
                
                output_32k[start:end] += out * chunk_weight
                weight[start:end] += chunk_weight
                
                # Move to next chunk
                start += chunk_samples - overlap_samples
            
            # Normalize by weights
            weight = weight.clamp(min=1e-8)
            output_32k = output_32k / weight
        
        # Resample back to 44.1kHz
        output_44k = self._resample(output_32k.unsqueeze(0), CLAPSEP_SAMPLE_RATE, SAMPLE_RATE).squeeze(0)
        
        # Ensure same length as input
        if output_44k.shape[-1] != original_length:
            if output_44k.shape[-1] > original_length:
                output_44k = output_44k[:original_length]
            else:
                output_44k = F.pad(output_44k, (0, original_length - output_44k.shape[-1]))
        
        # Convert back to stereo by duplicating
        output_stereo = output_44k.unsqueeze(0).repeat(2, 1)  # (2, T)
        
        return output_stereo.cpu()
    
    def separate_all(self, mixture: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not self._available:
            raise RuntimeError("CLAPSep model not available")
        
        results = {}
        for stem in STEMS:
            results[stem] = self.separate(mixture, stem)
        return results


# ============================================================================
# Evaluation Functions
# ============================================================================

def compute_sdr(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    """
    Compute SDR between estimate and reference.
    
    Args:
        estimate: (C, T) estimated waveform
        reference: (C, T) reference waveform
        
    Returns:
        SDR in dB
    """
    # Use our existing SDR function (returns negative for loss, so negate)
    estimate = estimate.unsqueeze(0)  # (1, C, T)
    reference = reference.unsqueeze(0)  # (1, C, T)
    sdr = -sdr_loss(estimate, reference).item()
    return sdr


def compute_sisdr(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    """
    Compute SI-SDR (Scale-Invariant SDR) between estimate and reference.
    
    Args:
        estimate: (C, T) estimated waveform
        reference: (C, T) reference waveform
        
    Returns:
        SI-SDR in dB
    """
    # Use our existing SI-SDR function (returns negative for loss, so negate)
    estimate = estimate.unsqueeze(0)  # (1, C, T)
    reference = reference.unsqueeze(0)  # (1, C, T)
    sisdr = -sisdr_loss(estimate, reference).item()
    return sisdr


def load_track_stems(filepath: Path) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Load mixture and all ground truth stems from a .stem.mp4 file.
    
    Returns:
        mixture: (C, T) tensor
        stems: Dict mapping stem names to (C, T) tensors
    """
    import stempeg
    
    stems_np, rate = stempeg.read_stems(str(filepath))
    # stems_np shape: (num_stems, samples, channels) = (5, T, 2)
    # Order: [mixture, drums, bass, other, vocals]
    
    stems_tensor = torch.from_numpy(stems_np).permute(0, 2, 1).float()  # (5, C, T)
    
    mixture = stems_tensor[0]  # (C, T)
    stem_dict = {
        "drums": stems_tensor[1],
        "bass": stems_tensor[2],
        "other": stems_tensor[3],
        "vocals": stems_tensor[4],
    }
    
    return mixture, stem_dict


@dataclass
class TrackResult:
    """Results for a single track."""
    track_name: str
    model_name: str
    # SDR metrics
    sdr_drums: float
    sdr_bass: float
    sdr_other: float
    sdr_vocals: float
    sdr_avg: float
    # SI-SDR metrics
    sisdr_drums: float
    sisdr_bass: float
    sisdr_other: float
    sisdr_vocals: float
    sisdr_avg: float


def evaluate_model_on_track(
    model: SeparationModel,
    mixture: torch.Tensor,
    reference_stems: Dict[str, torch.Tensor],
    track_name: str,
    device: str = "cuda",
    plot_spectrograms: bool = False,
    use_wandb: bool = False,
    output_dir: Optional[Path] = None,
) -> Tuple[TrackResult, Optional[Dict[str, torch.Tensor]]]:
    """
    Evaluate a model on a single track.
    
    Returns:
        Tuple of (TrackResult, estimated_stems dict if plot_spectrograms else None)
    """
    
    # Separate all stems
    estimated_stems = model.separate_all(mixture)
    
    # Compute SDR and SI-SDR for each stem
    sdr_scores = {}
    sisdr_scores = {}
    for stem in STEMS:
        estimate = estimated_stems[stem].cpu()
        reference = reference_stems[stem]
        
        # Ensure same length
        min_len = min(estimate.shape[-1], reference.shape[-1])
        estimate = estimate[:, :min_len]
        reference = reference[:, :min_len]
        
        sdr_scores[stem] = compute_sdr(estimate, reference)
        sisdr_scores[stem] = compute_sisdr(estimate, reference)
    
    # Compute averages
    sdr_avg = sum(sdr_scores.values()) / len(sdr_scores)
    sisdr_avg = sum(sisdr_scores.values()) / len(sisdr_scores)
    
    result = TrackResult(
        track_name=track_name,
        model_name=model.name,
        sdr_drums=sdr_scores["drums"],
        sdr_bass=sdr_scores["bass"],
        sdr_other=sdr_scores["other"],
        sdr_vocals=sdr_scores["vocals"],
        sdr_avg=sdr_avg,
        sisdr_drums=sisdr_scores["drums"],
        sisdr_bass=sisdr_scores["bass"],
        sisdr_other=sisdr_scores["other"],
        sisdr_vocals=sisdr_scores["vocals"],
        sisdr_avg=sisdr_avg,
    )
    
    # Generate and log spectrograms if requested
    if plot_spectrograms:
        # Prepare estimated stems with correct lengths
        estimated_for_plot = {}
        reference_for_plot = {}
        for stem in STEMS:
            est = estimated_stems[stem].cpu()
            ref = reference_stems[stem]
            min_len = min(est.shape[-1], ref.shape[-1])
            estimated_for_plot[stem] = est[:, :min_len]
            reference_for_plot[stem] = ref[:, :min_len]
        
        # Trim mixture too
        min_len = min(mixture.shape[-1], min(est.shape[-1] for est in estimated_for_plot.values()))
        mixture_trimmed = mixture[:, :min_len]
        
        if use_wandb:
            import wandb
            # Log to W&B
            log_all_stems_to_wandb(
                mixture=mixture_trimmed,
                estimated_stems=estimated_for_plot,
                reference_stems=reference_for_plot,
                step=None,  # No step for benchmark
                sample_rate=SAMPLE_RATE,
                log_individual=True,
                log_combined=True,
            )
            # Also log track name as context
            wandb.log({f"track/{track_name}/model": model.name})
        
        # Save spectrograms to disk if output_dir provided
        if output_dir is not None:
            spec_dir = output_dir / "spectrograms" / track_name.replace(" ", "_")
            spec_dir.mkdir(parents=True, exist_ok=True)
            
            # Save combined plot
            fig = plot_all_stems_spectrograms(
                mixture=mixture_trimmed,
                estimated_stems=estimated_for_plot,
                reference_stems=reference_for_plot,
                sample_rate=SAMPLE_RATE,
            )
            model_name_safe = model.name.replace(" ", "_").replace("(", "").replace(")", "")
            fig.savefig(spec_dir / f"{model_name_safe}_all_stems.png", dpi=150, bbox_inches='tight')
            import matplotlib.pyplot as plt
            plt.close(fig)
    
    return result, estimated_stems if plot_spectrograms else None


def evaluate_model(
    model: SeparationModel,
    test_files: List[Path],
    device: str = "cuda",
    plot_spectrograms: bool = False,
    use_wandb: bool = False,
    output_dir: Optional[Path] = None,
) -> List[TrackResult]:
    """Evaluate a model on all test tracks."""
    
    results = []
    
    print(f"\nEvaluating {model.name}...")
    for filepath in tqdm(test_files, desc=model.name):
        track_name = filepath.stem.replace(".stem", "")
        
        try:
            mixture, reference_stems = load_track_stems(filepath)
            result, _ = evaluate_model_on_track(
                model, mixture, reference_stems, track_name, device,
                plot_spectrograms=plot_spectrograms,
                use_wandb=use_wandb,
                output_dir=output_dir,
            )
            results.append(result)
            
            # Print per-track results
            print(f"  {track_name}:")
            print(f"    SDR:   avg={result.sdr_avg:.2f} dB "
                  f"(D={result.sdr_drums:.1f}, B={result.sdr_bass:.1f}, "
                  f"O={result.sdr_other:.1f}, V={result.sdr_vocals:.1f})")
            print(f"    SISDR: avg={result.sisdr_avg:.2f} dB "
                  f"(D={result.sisdr_drums:.1f}, B={result.sisdr_bass:.1f}, "
                  f"O={result.sisdr_other:.1f}, V={result.sisdr_vocals:.1f})")
            
        except Exception as e:
            print(f"  Error processing {track_name}: {e}")
            continue
    
    return results


def aggregate_results(results: List[TrackResult]) -> Dict[str, Dict[str, float]]:
    """Aggregate results across all tracks."""
    if not results:
        return {}
    
    return {
        "sdr": {
            "drums": np.mean([r.sdr_drums for r in results]),
            "bass": np.mean([r.sdr_bass for r in results]),
            "other": np.mean([r.sdr_other for r in results]),
            "vocals": np.mean([r.sdr_vocals for r in results]),
            "average": np.mean([r.sdr_avg for r in results]),
        },
        "sisdr": {
            "drums": np.mean([r.sisdr_drums for r in results]),
            "bass": np.mean([r.sisdr_bass for r in results]),
            "other": np.mean([r.sisdr_other for r in results]),
            "vocals": np.mean([r.sisdr_vocals for r in results]),
            "average": np.mean([r.sisdr_avg for r in results]),
        },
    }


def print_results_table(all_results: Dict[str, List[TrackResult]]):
    """Print a formatted results table."""
    
    # SDR Table
    print("\n" + "=" * 85)
    print("EVALUATION RESULTS - SDR (Signal-to-Distortion Ratio)")
    print("=" * 85)
    print(f"{'Model':<35} {'Drums':>9} {'Bass':>9} {'Other':>9} {'Vocals':>9} {'Avg':>9}")
    print("-" * 85)
    
    for model_name, results in all_results.items():
        agg = aggregate_results(results)
        if agg:
            sdr = agg['sdr']
            print(f"{model_name:<35} "
                  f"{sdr['drums']:>9.2f} "
                  f"{sdr['bass']:>9.2f} "
                  f"{sdr['other']:>9.2f} "
                  f"{sdr['vocals']:>9.2f} "
                  f"{sdr['average']:>9.2f}")
    
    print("=" * 85)
    
    # SI-SDR Table
    print("\n" + "=" * 85)
    print("EVALUATION RESULTS - SI-SDR (Scale-Invariant SDR)")
    print("=" * 85)
    print(f"{'Model':<35} {'Drums':>9} {'Bass':>9} {'Other':>9} {'Vocals':>9} {'Avg':>9}")
    print("-" * 85)
    
    for model_name, results in all_results.items():
        agg = aggregate_results(results)
        if agg:
            sisdr = agg['sisdr']
            print(f"{model_name:<35} "
                  f"{sisdr['drums']:>9.2f} "
                  f"{sisdr['bass']:>9.2f} "
                  f"{sisdr['other']:>9.2f} "
                  f"{sisdr['vocals']:>9.2f} "
                  f"{sisdr['average']:>9.2f}")
    
    print("=" * 85)
    print("All values in dB (higher is better)")
    print()


def save_results(all_results: Dict[str, List[TrackResult]], output_dir: Path):
    """Save results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable format
    output = {}
    for model_name, results in all_results.items():
        output[model_name] = {
            "per_track": [
                {
                    "track": r.track_name,
                    "sdr": {
                        "drums": r.sdr_drums,
                        "bass": r.sdr_bass,
                        "other": r.sdr_other,
                        "vocals": r.sdr_vocals,
                        "average": r.sdr_avg,
                    },
                    "sisdr": {
                        "drums": r.sisdr_drums,
                        "bass": r.sisdr_bass,
                        "other": r.sisdr_other,
                        "vocals": r.sisdr_vocals,
                        "average": r.sisdr_avg,
                    },
                }
                for r in results
            ],
            "aggregate": aggregate_results(results),
        }
    
    output_file = output_dir / "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to {output_file}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate stem separation models on MusDB18")
    parser.add_argument("--test-dir", type=str, default="../musdb18/test",
                        help="Path to MusDB18 test directory")
    parser.add_argument("--checkpoint", type=str, default="/Users/surya/Downloads/2025_12_01_batch4/best_model.pt",
                        help="Path to our model checkpoint")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (auto-detect if not specified)")
    parser.add_argument("--models", type=str, nargs="+", default=["ours", "htdemucs", "clapsep"],
                        choices=["ours", "htdemucs", "clapsep"],
                        help="Models to evaluate")
    parser.add_argument("--max-tracks", type=int, default=None,
                        help="Maximum number of tracks to evaluate (for quick testing)")
    parser.add_argument("--plot-spectrograms", action="store_true",
                        help="Generate and save spectrograms for visualization")
    parser.add_argument("--use-wandb", action="store_true",
                        help="Log spectrograms to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="audio-text-htdemucs-benchmark",
                        help="W&B project name (only used if --use-wandb is set)")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="W&B run name (optional, auto-generated if not set)")
    
    args = parser.parse_args()
    
    # Set device (with MPS support for Apple Silicon)
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Find test files
    test_dir = Path(args.test_dir)
    test_files = sorted(test_dir.glob("*.stem.mp4"))
    
    if not test_files:
        raise ValueError(f"No .stem.mp4 files found in {test_dir}")
    
    if args.max_tracks:
        test_files = test_files[:args.max_tracks]
    
    print(f"Found {len(test_files)} test tracks")
    
    # Initialize W&B if requested
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "test_dir": args.test_dir,
                "checkpoint": args.checkpoint,
                "models": args.models,
                "num_tracks": len(test_files),
                "plot_spectrograms": args.plot_spectrograms,
            }
        )
        print(f"Initialized W&B project: {args.wandb_project}")
    
    # Load models
    models = {}
    
    if "ours" in args.models:
        models["AudioTextHTDemucs (Ours)"] = OurModel(args.checkpoint, device)
    
    if "htdemucs" in args.models:
        models["HTDemucs (Baseline)"] = HTDemucsModel(device)
    
    if "clapsep" in args.models:
        clapsep = CLAPSepModel(device)
        if clapsep.is_available():
            models["CLAPSep (Baseline)"] = clapsep
        else:
            print("Skipping CLAPSep - model not available")
    
    # Evaluate each model
    all_results = {}
    output_dir = Path(args.output_dir)
    
    for model_name, model in models.items():
        results = evaluate_model(
            model, test_files, device,
            plot_spectrograms=args.plot_spectrograms,
            use_wandb=args.use_wandb,
            output_dir=output_dir if args.plot_spectrograms else None,
        )
        all_results[model_name] = results
    
    # Print and save results
    print_results_table(all_results)
    save_results(all_results, output_dir)
    
    # Log final aggregate metrics to W&B
    if args.use_wandb:
        import wandb
        for model_name, results in all_results.items():
            agg = aggregate_results(results)
            if agg:
                prefix = model_name.replace(" ", "_").replace("(", "").replace(")", "")
                wandb.log({
                    f"final/{prefix}/sdr_avg": agg["sdr"]["average"],
                    f"final/{prefix}/sisdr_avg": agg["sisdr"]["average"],
                })
        wandb.finish()
        print("W&B run finished.")


if __name__ == "__main__":
    main()

