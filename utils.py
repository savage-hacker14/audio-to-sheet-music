
from typing import Union, Optional, Dict, List
from pathlib import Path
import yaml

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/training use


# ============================================================================
# YAML Config
# ============================================================================

def load_config(file_path: Union[str, Path]) -> dict:
    """Load a YAML configuration file."""
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
        
    return config


# ============================================================================
# Spectrogram Utilities
# ============================================================================

def compute_spectrogram(
    waveform: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 512,
    power: float = 2.0,
    to_db: bool = True,
    top_db: float = 80.0,
) -> torch.Tensor:
    """
    Compute spectrogram from waveform using STFT.
    
    Args:
        waveform: (C, T) or (T,) audio waveform
        n_fft: FFT window size
        hop_length: Hop length between frames
        power: Exponent for magnitude (1.0 for magnitude, 2.0 for power)
        to_db: Convert to decibel scale
        top_db: Threshold for dynamic range in dB
        
    Returns:
        (F, T') spectrogram tensor
    """
    # Handle stereo by taking mean to mono
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)  # (T,)
    
    # Move to CPU for STFT computation
    waveform = waveform.cpu()
    
    # Compute STFT
    window = torch.hann_window(n_fft)
    stft = torch.stft(
        waveform, 
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
        center=True,
        pad_mode='reflect'
    )
    
    # Compute magnitude spectrogram
    spec = torch.abs(stft).pow(power)
    
    # Convert to dB
    if to_db:
        spec = amplitude_to_db(spec, top_db=top_db)
    
    return spec


def amplitude_to_db(
    spec: torch.Tensor,
    ref: float = 1.0,
    amin: float = 1e-10,
    top_db: float = 80.0,
) -> torch.Tensor:
    """Convert amplitude/power spectrogram to decibel scale."""
    spec_db = 10.0 * torch.log10(torch.clamp(spec, min=amin) / ref)
    
    # Clip to top_db range
    max_val = spec_db.max()
    spec_db = torch.clamp(spec_db, min=max_val - top_db)
    
    return spec_db


def plot_spectrogram(
    spec: torch.Tensor,
    sample_rate: int = 44100,
    hop_length: int = 512,
    title: str = "Spectrogram",
    figsize: tuple = (10, 4),
    cmap: str = "magma",
    colorbar: bool = True,
) -> plt.Figure:
    """
    Plot a single spectrogram.
    
    Args:
        spec: (F, T) spectrogram tensor (in dB scale)
        sample_rate: Audio sample rate
        hop_length: Hop length used for STFT
        title: Plot title
        figsize: Figure size
        cmap: Colormap for spectrogram
        colorbar: Whether to show colorbar
        
    Returns:
        matplotlib Figure object
    """
    spec_np = spec.detach().cpu().numpy() if isinstance(spec, torch.Tensor) else spec
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute time and frequency axes
    n_frames = spec_np.shape[1]
    n_freqs = spec_np.shape[0]
    time_max = n_frames * hop_length / sample_rate
    freq_max = sample_rate / 2  # Nyquist frequency
    
    img = ax.imshow(
        spec_np,
        aspect='auto',
        origin='lower',
        cmap=cmap,
        extent=[0, time_max, 0, freq_max / 1000]  # freq in kHz
    )
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (kHz)')
    ax.set_title(title)
    
    if colorbar:
        cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
        cbar.set_label('Magnitude (dB)')
    
    fig.tight_layout()
    return fig


def plot_spectrogram_comparison(
    spectrograms: Dict[str, torch.Tensor],
    sample_rate: int = 44100,
    hop_length: int = 512,
    figsize: tuple = (14, 3),
    cmap: str = "magma",
    suptitle: Optional[str] = None,
) -> plt.Figure:
    """
    Plot multiple spectrograms side by side for comparison.
    
    Args:
        spectrograms: Dict mapping names to spectrogram tensors
        sample_rate: Audio sample rate
        hop_length: Hop length used for STFT
        figsize: Figure size (width, height per row)
        cmap: Colormap for spectrograms
        suptitle: Super title for the figure
        
    Returns:
        matplotlib Figure object
    """
    n_specs = len(spectrograms)
    fig, axes = plt.subplots(1, n_specs, figsize=(figsize[0], figsize[1]))
    
    if n_specs == 1:
        axes = [axes]
    
    # Find global min/max for consistent colorbar
    all_specs = [s.detach().cpu().numpy() if isinstance(s, torch.Tensor) else s 
                 for s in spectrograms.values()]
    vmin = min(s.min() for s in all_specs)
    vmax = max(s.max() for s in all_specs)
    
    for ax, (name, spec) in zip(axes, spectrograms.items()):
        spec_np = spec.detach().cpu().numpy() if isinstance(spec, torch.Tensor) else spec
        
        n_frames = spec_np.shape[1]
        time_max = n_frames * hop_length / sample_rate
        freq_max = sample_rate / 2
        
        img = ax.imshow(
            spec_np,
            aspect='auto',
            origin='lower',
            cmap=cmap,
            extent=[0, time_max, 0, freq_max / 1000],
            vmin=vmin,
            vmax=vmax,
        )
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (kHz)')
        ax.set_title(name)
    
    # Add single colorbar
    fig.colorbar(img, ax=axes, format='%+2.0f dB', label='Magnitude (dB)')
    
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=1.02)
    
    fig.tight_layout()
    return fig


def plot_separation_spectrograms(
    mixture: torch.Tensor,
    estimated: torch.Tensor,
    reference: torch.Tensor,
    stem_name: str = "stem",
    sample_rate: int = 44100,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> plt.Figure:
    """
    Create a comparison spectrogram plot for stem separation.
    Shows mixture, estimated, reference, and difference.
    
    Args:
        mixture: (C, T) mixture waveform
        estimated: (C, T) estimated stem waveform
        reference: (C, T) ground truth stem waveform
        stem_name: Name of the stem for title
        sample_rate: Audio sample rate
        n_fft: FFT window size
        hop_length: Hop length
        
    Returns:
        matplotlib Figure object
    """
    # Compute spectrograms
    spec_mix = compute_spectrogram(mixture, n_fft=n_fft, hop_length=hop_length)
    spec_est = compute_spectrogram(estimated, n_fft=n_fft, hop_length=hop_length)
    spec_ref = compute_spectrogram(reference, n_fft=n_fft, hop_length=hop_length)
    
    # Create comparison plot
    spectrograms = {
        "Mixture": spec_mix,
        f"Estimated ({stem_name})": spec_est,
        f"Ground Truth ({stem_name})": spec_ref,
    }
    
    fig = plot_spectrogram_comparison(
        spectrograms,
        sample_rate=sample_rate,
        hop_length=hop_length,
        suptitle=f"Stem Separation: {stem_name.capitalize()}"
    )
    
    return fig


def plot_all_stems_spectrograms(
    mixture: torch.Tensor,
    estimated_stems: Dict[str, torch.Tensor],
    reference_stems: Dict[str, torch.Tensor],
    sample_rate: int = 44100,
    n_fft: int = 2048,
    hop_length: int = 512,
    figsize: tuple = (16, 12),
) -> plt.Figure:
    """
    Create a grid of spectrograms for all stems.
    
    Args:
        mixture: (C, T) mixture waveform
        estimated_stems: Dict mapping stem names to estimated (C, T) waveforms
        reference_stems: Dict mapping stem names to reference (C, T) waveforms
        sample_rate: Audio sample rate
        n_fft: FFT window size
        hop_length: Hop length
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    stem_names = list(estimated_stems.keys())
    n_stems = len(stem_names)
    
    # Create grid: rows = stems, cols = [Estimated, Ground Truth]
    fig, axes = plt.subplots(n_stems, 2, figsize=figsize)
    
    if n_stems == 1:
        axes = axes.reshape(1, -1)
    
    # Compute mixture spectrogram for reference
    spec_mix = compute_spectrogram(mixture, n_fft=n_fft, hop_length=hop_length)
    
    for row, stem_name in enumerate(stem_names):
        # Estimated
        spec_est = compute_spectrogram(
            estimated_stems[stem_name], n_fft=n_fft, hop_length=hop_length
        )
        # Reference
        spec_ref = compute_spectrogram(
            reference_stems[stem_name], n_fft=n_fft, hop_length=hop_length
        )
        
        # Get time extent
        n_frames = spec_est.shape[1]
        time_max = n_frames * hop_length / sample_rate
        freq_max = sample_rate / 2
        
        # Plot estimated
        spec_np = spec_est.detach().cpu().numpy()
        axes[row, 0].imshow(
            spec_np, aspect='auto', origin='lower', cmap='magma',
            extent=[0, time_max, 0, freq_max / 1000]
        )
        axes[row, 0].set_title(f'{stem_name.capitalize()} - Estimated')
        axes[row, 0].set_ylabel('Freq (kHz)')
        
        # Plot reference
        spec_np = spec_ref.detach().cpu().numpy()
        img = axes[row, 1].imshow(
            spec_np, aspect='auto', origin='lower', cmap='magma',
            extent=[0, time_max, 0, freq_max / 1000]
        )
        axes[row, 1].set_title(f'{stem_name.capitalize()} - Ground Truth')
        
    # Set x labels on bottom row
    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Time (s)')
    
    fig.colorbar(img, ax=axes, format='%+2.0f dB', label='Magnitude (dB)')
    fig.suptitle('Stem Separation Results', fontsize=14, y=1.0)
    fig.tight_layout()
    
    return fig


# ============================================================================
# Weights & Biases Logging Utilities
# ============================================================================

def log_spectrogram_to_wandb(
    fig: plt.Figure,
    key: str = "spectrogram",
    step: Optional[int] = None,
    caption: Optional[str] = None,
):
    """
    Log a matplotlib figure as an image to W&B.
    
    Args:
        fig: matplotlib Figure object
        key: W&B log key
        step: Training step (optional)
        caption: Image caption
    """
    import wandb
    
    # Convert figure to W&B Image
    wandb_img = wandb.Image(fig, caption=caption)
    
    log_dict = {key: wandb_img}
    if step is not None:
        wandb.log(log_dict, step=step)
    else:
        wandb.log(log_dict)
    
    # Close the figure to free memory
    plt.close(fig)

def log_audio_to_wandb(
    audio: torch.Tensor,
    stem_name: str,
    is_gt: bool,
    sample_rate: int = 44100
):
    """
    Log audio waveform to W&B.
    
    Args:
        audio: (C, T) audio waveform tensor
        stem_name: Name of the stem
        is_gt: Whether this is ground truth audio (or extracted audio)
        sample_rate: Audio sample rate
    """
    import wandb
    
    # Convert to numpy
    audio_np = audio.detach().cpu().numpy().T  # (T, C)
    title =f"true_{stem_name}" if is_gt else f"extracted_{stem_name}"
    keyname = f"audio/{title}"
    wandb.log({
        keyname: wandb.Audio(
            audio_np,
            sample_rate=sample_rate,
            caption=title
        )
    })

def log_separation_spectrograms_to_wandb(
    mixture: torch.Tensor,
    estimated: torch.Tensor,
    reference: torch.Tensor,
    stem_name: str,
    step: Optional[int] = None,
    sample_rate: int = 44100,
):
    """
    Log stem separation spectrograms to W&B.
    
    Args:
        mixture: (C, T) mixture waveform
        estimated: (C, T) estimated stem waveform
        reference: (C, T) ground truth stem waveform
        stem_name: Name of the stem
        step: Training step (optional)
        sample_rate: Audio sample rate
    """
    fig = plot_separation_spectrograms(
        mixture=mixture,
        estimated=estimated,
        reference=reference,
        stem_name=stem_name,
        sample_rate=sample_rate,
    )
    
    log_spectrogram_to_wandb(
        fig=fig,
        key=f"spectrograms/{stem_name}",
        step=step,
        caption=f"Separation for {stem_name}"
    )


def log_all_stems_to_wandb(
    mixture: torch.Tensor,
    estimated_stems: Dict[str, torch.Tensor],
    reference_stems: Dict[str, torch.Tensor],
    step: Optional[int] = None,
    sample_rate: int = 44100,
    log_individual: bool = True,
    log_combined: bool = True,
):
    """
    Log spectrograms for all stems to W&B.
    
    Args:
        mixture: (C, T) mixture waveform
        estimated_stems: Dict mapping stem names to estimated (C, T) waveforms
        reference_stems: Dict mapping stem names to reference (C, T) waveforms
        step: Training step (optional)
        sample_rate: Audio sample rate
        log_individual: Log individual stem comparisons
        log_combined: Log combined grid of all stems
    """
    if log_individual:
        for stem_name in estimated_stems.keys():
            log_separation_spectrograms_to_wandb(
                mixture=mixture,
                estimated=estimated_stems[stem_name],
                reference=reference_stems[stem_name],
                stem_name=stem_name,
                step=step,
                sample_rate=sample_rate,
            )
    
    if log_combined:
        fig = plot_all_stems_spectrograms(
            mixture=mixture,
            estimated_stems=estimated_stems,
            reference_stems=reference_stems,
            sample_rate=sample_rate,
        )
        log_spectrogram_to_wandb(
            fig=fig,
            key="spectrograms/all_stems",
            step=step,
            caption="All stems separation comparison"
        )

# --- Audio I/O ---

# def load_audio(
#     file_path: Union[str, Path],
#     sample_rate: int = DEFAULT_SAMPLE_RATE,
#     max_len: int = 5,
#     mono: bool = True
# ) -> Tuple[np.ndarray, int]:
#     """
#     Load an audio file into a numpy array.

#     Parameters
#     ----------
#     file_path (str or Path): Path to the audio file
#     max_len (int): Maximum length of audio in seconds
#     sample_rate (int, optional): Target sample rate
#     mono (bool, optional): Whether to convert audio to mono

#     Returns
#     -------
#     tuple
#         (audio_data, sample_rate)
#     """
#     try:
#         audio_data, sr = librosa.load(file_path, sr=sample_rate, mono=mono)
        
#         # Clip audio to max_len
#         max_samples = int(sample_rate * max_len)
#         if len(audio_data) > max_samples:
#             audio_data = audio_data[:max_samples]
#         else:
#             padding = max_samples - len(audio_data)
#             audio_data = np.pad(
#                 audio_data, 
#                 (0, padding), 
#                 'constant'
#             )
            
#         return audio_data, sr
#     except Exception as e:
#         raise IOError(f"Error loading audio file {file_path}: {str(e)}")

# def save_audio(
#     audio_data: np.ndarray,
#     file_path: Union[str, Path],
#     sample_rate: int = DEFAULT_SAMPLE_RATE,
#     normalize: bool = True,
#     file_format: str = 'flac'
# ) -> None:
#     """
#     Save audio data to a file.

#     Parameters
#     ----------
#     audio_data   (np.ndarray): Audio time series
#     file_path    (str or Path): Path to save the audio file
#     sample_rate  (int, optional): Sample rate of audio
#     normalize    (bool, optional): Whether to normalize audio before saving
#     file_format  (str, optional): Audio file format
    
#     Returns
#     -------
#     None
#     """
#     output_dir = Path(file_path).parent
#     if output_dir and not output_dir.exists():
#         try:
#             output_dir.mkdir(parents=True, exist_ok=True)
#         except Exception as e:
#             raise IOError(f"Error creating directory {output_dir}: {str(e)}")
        
#     # Normalize audio before saving
#     audio_data = librosa.util.normalize(audio_data) if normalize else audio_data
    
#     try:
#         sf.write(file_path, audio_data, sample_rate, format=file_format)
#     except Exception as e:
#         raise IOError(f"Error saving audio to {file_path}: {str(e)}")

# # --- Gap Processing ---

# def create_gap_mask(
#     audio_len_samples: int,
#     gap_len_s: float,
#     sample_rate: int = DEFAULT_SAMPLE_RATE,
#     gap_start_s: Optional[float] = None,
# ) -> Tuple[np.ndarray, Tuple[int, int]]:
#     """
#     Creates a binary mask with a single gap of zeros at a random location.

#     Parameters
#     ----------
#     audio_len_samples : int
#         Length of the target audio in samples.
#     gap_len_s : float
#         Desired gap length in seconds.
#     sample_rate : int, optional
#         Sample rate. Defaults to DEFAULT_SAMPLE_RATE.
#     gap_start_s : float, optional
#         Timestap in seconds where the gap starts. If None, a random position is chosen.

#     Returns
#     -------
#     Tuple[np.ndarray, Tuple[int, int]]
#         (mask, (gap_start_sample, gap_end_sample))
#         Mask is 1.0 for signal, 0.0 for gap (float32).
#         Interval is gap start/end indices in samples.
#     """
#     gap_len_samples = int(gap_len_s * sample_rate)

#     if gap_len_samples <= 0:
#         # No gap, return full mask and zero interval
#         return np.ones(audio_len_samples, dtype=np.float32), (0, 0)

#     if gap_len_samples >= audio_len_samples:
#         # Gap covers everything
#         print(f"Warning: Gap length ({gap_len_s}s) >= audio length. Returning all zeros mask.")
#         return np.zeros(audio_len_samples, dtype=np.float32), (0, audio_len_samples)

#     # Choose a random start position for the gap (inclusive range)
#     max_start_sample = audio_len_samples - gap_len_samples
#     if (gap_start_s is None):
#         gap_start_sample = np.random.randint(0, max_start_sample + 1)
#     else:
#         gap_start_sample = int(gap_start_s * sample_rate)

#     gap_end_sample = gap_start_sample + gap_len_samples

#     # Create mask
#     mask = np.ones(audio_len_samples, dtype=np.float32)
#     mask[gap_start_sample:gap_end_sample] = 0.0

#     return mask, (gap_start_sample, gap_end_sample)

# def add_random_gap(
#         file_path: Union[str, Path],
#         gap_len: int,
#         sample_rate: int = DEFAULT_SAMPLE_RATE,
#         mono: bool = True
# ) -> Tuple[np.ndarray, Tuple[float, float]]:
#     """
#     Add a random gap of length gap_len at a random valid position within the audio file and return the audio data
    
#     Parameters
#     ----------
#     file_path (str or Path): Path to the audio file
#     gap_len (int): Gap length (seconds) to add at one location within the audio file
#     sample_rate (int, optional): Target sample rate
#     mono (bool, optional): Whether to convert audio to mono

#     Returns
#     -------
#     tuple
#         (modified_audio_data, gap_interval)
#         gap_interval is a tuple of (start_time, end_time) in seconds
#     """
#     audio_data, sr = load_audio(file_path, sample_rate=sample_rate, mono=mono)
    
#     # Convert gap length to samples
#     gap_length    = int(gap_len * sample_rate)
#     audio_len     = len(audio_data)
    
#     # Handle case where gap is longer than audio
#     if gap_length >= audio_len:
#         raise ValueError(f"Gap length ({gap_length}s) exceeds audio length ({audio_len/sample_rate}s)")
    
#     # Get sample indices for gap placement
#     gap_start_idx = np.random.randint(0, audio_len - int(gap_len * sample_rate))
#     silence       = np.zeros(gap_length)

#     # Add gap
#     audio_new = np.concatenate([audio_data[:gap_start_idx], silence, audio_data[gap_start_idx + gap_length:]])

#     # Return gap interval as a tuple
#     gap_interval = (gap_start_idx / sample_rate, (gap_start_idx + gap_length) / sample_rate)

#     return audio_new, gap_interval
  
# # --- STFT Processing ---

# def extract_spectrogram(
#     audio_data: np.ndarray,
#     n_fft: int = 2048,
#     hop_length: int = 512,
#     win_length: Optional[int] = None,
#     window: str = 'hann',
#     center: bool = True,
#     power: float = 1.0
# ) -> np.ndarray:
#     """
#     Extract magnitude spectrogram from audio data.

#     Parameters
#     ----------
#     audio_data (np.ndarray): Audio time series
#     n_fft (int, optional): FFT window size
#     hop_length (int, optional): Number of samples between successive frames
#     win_length (int or None, optional): Window length. If None, defaults to n_fft
#     window (str, optional): Window specification
#     center (bool, optional): If True, pad signal on both sides
#     power (float, optional): Exponent for the magnitude spectrogram (e.g. 1 for energy, 2 for power)
    
#     Returns
#     -------
#     np.ndarray
#         Magnitude spectrogram
#     """
#     if power < 0:
#         raise ValueError("Power must be non-negative")
    
#     if win_length is None:
#         win_length = n_fft
    
#     stft = librosa.stft(
#         audio_data,
#         n_fft=n_fft,
#         hop_length=hop_length,
#         win_length=win_length,
#         window=window,
#         center=center
#     )
    
#     return stft

# def extract_mel_spectrogram(
#     audio_data: np.ndarray,
#     sample_rate: int = DEFAULT_SAMPLE_RATE,
#     n_fft: int = 2048,
#     hop_length: int = 512,
#     n_mels: int = 128,
#     fmin: float = 0.0,
#     fmax: Optional[float] = None,
#     power: float = 2.0
# ) -> np.ndarray:
#     """
#     Extract mel spectrogram from audio data.

#     Parameters
#     ----------
#     audio_data (np.ndarray): Audio time series
#     sample_rate (int, optional): Sample rate of audio
#     n_fft (int, optional): FFT window size
#     hop_length (int, optional): Number of samples between successive frames
#     n_mels (int, optional): Number of mel bands
#     fmin (float, optional): Minimum frequency
#     fmax (float or None, optional): Maximum frequency. If None, use sample_rate/2
#     power (float, optional): Exponent for the magnitude spectrogram (e.g. 1 for energy, 2 for power)

#     Returns
#     -------
#     np.ndarray
#         Mel spectrogram
#     """
#     if power < 0:
#         raise ValueError("Power must be non-negative")
    
#     return librosa.feature.melspectrogram(
#         y=audio_data,
#         sr=sample_rate,
#         n_fft=n_fft,
#         hop_length=hop_length,
#         n_mels=n_mels,
#         fmin=fmin,
#         fmax=fmax,
#         power=power
#     )

# def spectrogram_to_audio(
#     spectrogram: np.ndarray,
#     phase: Optional[np.ndarray] = None,
#     phase_info: bool = False,
#     n_fft=512,
#     n_iter=64,
#     window='hann',
#     hop_length=512,
#     win_length=None,
#     center=True) -> np.ndarray:
#     """
#     Convert a spectrogram back to audio using either:
#     1. Original phase information (if provided)
#     2. Griffin-Lim algorithm to estimate phase (if no phase provided)
    
#     Even with original phase, the reconstruction is not truely lossless 1e-33 MSE loss.
    
#     Parameters:
#     -----------
#     spectrogram (np.ndarray): The magnitude spectrogram to convert back to audio
#     phase       (np.ndarray, optional): Phase information to use for reconstruction. If None, Griffin-Lim is used.
#     phase_info  (bool): If True, the input is assumed to be a phase spectrogram
#     n_fft       (int): FFT window size
#     n_iter      (int, optional): Number of iterations for Griffin-Lim algorithm
#     window      (str): Window function to use
#     win_length  (int or None): Window size. If None, defaults to n_fft 
#     hop_length  (int, optional): Number of samples between successive frames
#     center      (bool, optional): Whether to pad the signal at the edges
         
#     Returns:
#     --------
#     y : np.ndarray The reconstructed audio signal
#     """
#     # If the input is in dB scale, convert back to amplitude
#     if np.max(spectrogram) < 0 and np.mean(spectrogram) < 0:
#         spectrogram = librosa.db_to_amplitude(spectrogram)
    
#     if phase_info:
#         return librosa.istft(spectrogram, n_fft=n_fft, hop_length=hop_length, 
#                           win_length=win_length, window=window, center=center)
    
#     # If phase information is provided, use it for reconstruction
#     if phase is not None:
#         # Combine magnitude and phase to form complex spectrogram
#         complex_spectrogram = spectrogram * np.exp(1j * phase)
        
#         # Inverse STFT to get audio
#         y = librosa.istft(complex_spectrogram, n_fft=n_fft, hop_length=hop_length, 
#                           win_length=win_length, window=window, center=center)
#     else:
#         # Use Griffin-Lim algorithm to estimate phase
#         y = librosa.griffinlim(spectrogram, n_fft=n_fft, n_iter=n_iter, 
#                                hop_length=hop_length, win_length=win_length, 
#                                window=window, center=center)
#     return y

# def mel_spectrogram_to_audio(
#     mel_spectrogram: np.ndarray,
#     sample_rate: int = DEFAULT_SAMPLE_RATE,
#     n_fft: int = 2048,
#     hop_length: int = 512,
#     n_iter: int = 32,
#     n_mels: int = 128,
#     fmin: float = 0.0,
#     fmax: Optional[float] = None,
#     power: float = 2.0
# ) -> np.ndarray:
#     """
#     Convert a mel spectrogram to audio using inverse transformation and Griffin-Lim.

#     Parameters
#     ----------
#     mel_spectrogram (np.ndarray): Mel spectrogram
#     sample_rate     (int, optional): Sample rate of audio
#     n_fft           (int, optional): FFT window size
#     hop_length      (int, optional): Number of samples between successive frames
#     n_iter          (int, optional): Number of iterations for Griffin-Lim
#     n_mels          (int, optional): Number of mel bands
#     fmin            (float, optional): Minimum frequency
#     fmax            (float or None, optional): Maximum frequency. If None, use sample_rate/2
#     power           (float, optional): Exponent for the magnitude spectrogram (e.g. 1 for energy, 2 for power)

#     Returns
#     -------
#     np.ndarray
#         Audio time series
#     """
#     # Create a mel filterbank
#     mel_basis = librosa.filters.mel(
#         sr=sample_rate, 
#         n_fft=n_fft,
#         n_mels=n_mels,
#         fmin=fmin,
#         fmax=fmax
#     )
    
#     # Compute the pseudo-inverse of the mel filterbank
#     mel_filterbank_inv = np.linalg.pinv(mel_basis) 

#     # Convert Mel spectrogram to linear spectrogram
#     linear_spec = np.dot(mel_filterbank_inv, mel_spectrogram)
    
#     # # If the input was a power spectrogram, take the square root
#     if power == 2.0:
#        linear_spec = np.sqrt(linear_spec)
    
#     # Perform Griffin-Lim to estimate the phase and convert to audio
#     audio_data = librosa.griffinlim(
#         linear_spec,
#         hop_length=hop_length,
#         n_fft=n_fft,
#         n_iter=n_iter
#     )
    
#     return audio_data

# def visualize_spectrogram(
#     spectrogram: np.ndarray,
#     power: int = 1,
#     sample_rate: int = DEFAULT_SAMPLE_RATE,
#     n_fft: int = 512,
#     hop_length: int = 192,
#     win_length: int = 384,
#     gap_int: Optional[Tuple[int, int]] = None,
#     in_db: bool = False,
#     y_axis: str = 'log',
#     x_axis: str = 'time',
#     title: str = 'Spectrogram',
#     save_path: Optional[Union[str, Path]] = None
# ) -> figure:
#     """
#     Visualize a spectrogram.

#     Parameters
#     ----------
#     spectrogram (np.ndarray): Spectrogram to visualize
#     power       (int): Whether the spectrogram is in energy (1) or power (2) scale
#     sample_rate (int, optional): Sample rate of audio
#     hop_length  (int, optional): Number of samples between successive frames
#     gap_int     (float tuple, optional): Start and end time [s] of the gap (if given) to be plotted as vertical lines
#     in_db       (bool, optional): Whether the spectrogram is already in dB scale
#     y_axis      (str, optional): Scale for the y-axis ('linear', 'log', or 'mel')
#     x_axis      (str, optional): Scale for the x-axis ('time' or 'frames')
#     title       (str, optional): Title for the plot
#     save_path   (str or Path or None, optional): Path to save the visualization. If None, the plot is displayed.
    
#     Returns
#     -------
#     Figure or None
#         The matplotlib Figure object if save_path is None, otherwise None
#     """
#     if power not in (1, 2):
#         raise ValueError("Power must be 1 (energy) or 2 (power)")
    
#     # Convert to dB scale if needed
#     if in_db:
#         spectrogram_data = np.array(spectrogram)
#     elif power == 1:
#         spectrogram_data = librosa.amplitude_to_db(spectrogram, ref=np.max, amin=1e-5, top_db=80)
#     else:  # power == 2
#         spectrogram_data = librosa.power_to_db(spectrogram, ref=np.max, amin=1e-5, top_db=80)
        

#     fig, ax = plt.subplots(figsize=(10, 4))
#     img = librosa.display.specshow(
#         spectrogram_data,
#         sr=sample_rate,
#         n_fft=n_fft,
#         win_length=win_length,
#         hop_length=hop_length,
#         y_axis=y_axis,
#         x_axis=x_axis,
#         ax=ax
#     )    

#     # Compute gap start and end indices and plot vertical lines
#     if gap_int is not None:
#         gap_start_s, gap_end_s = gap_int

#         ax.axvline(x=gap_start_s, color='white', linestyle='--', label='Gap Start')
#         ax.axvline(x=gap_end_s, color='white', linestyle='--', label='Gap End')
#         ax.legend()

#     # Add colorbar and title
#     fig.colorbar(img, ax=ax, format='%+2.0f dB')
#     ax.set_title(title)
#     fig.tight_layout()

#     # Save or return the figure
#     if save_path is not None:
#         save_path = Path(save_path)
#         output_dir = save_path.parent
#         if output_dir and not output_dir.exists():
#             output_dir.mkdir(parents=True, exist_ok=True)

#         fig.savefig(save_path)
#         plt.close(fig)
#         return None
    
#     return fig