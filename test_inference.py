import torch
import soundfile as sf
from pathlib import Path

from demucs import pretrained
from transformers import ClapModel, AutoTokenizer
from torch.utils.data import DataLoader
from torchaudio.transforms import Fade

from src.models.stem_separation.ATHTDemucs_v2 import AudioTextHTDemucs
from src.dataloader import MusDBStemDataset, collate_fn, STEM_PROMPTS
from src.loss import sdr_loss
from utils import plot_separation_spectrograms
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')  # Interactive backend

STEMS = ["drums", "bass", "other", "vocals"]
#STEMS = ["drums", "bass", "other", "vocals", "piano", "clapping", "guitar", "violin"]      # Adding negative query stems for testing

def load_model(checkpoint_path: str, device: str = "cuda") -> AudioTextHTDemucs:
    """Load model from checkpoint."""
    print("Loading pretrained HTDemucs...")
    htdemucs = pretrained.get_model('htdemucs').models[0]

    print("Loading CLAP model...")
    clap = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

    print("Building AudioTextHTDemucs model...")
    model = AudioTextHTDemucs(htdemucs, clap, tokenizer)

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def test_inference(
        checkpoint_path: str = "checkpoints/best_model.pt",
        data_dir: str = "data/quick_train",
        output_dir: str = "results",
        sample_rate: int = 44100,
        segment_seconds: float = 6.0,
        overlap: float = 0.1,
        device: str = None,
):
    """
    Run inference on a single song (extract all 4 stems), compute SDR for all stems, save results.

    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Directory containing a single .stem.mp4 file
        output_dir: Directory to save separated stems
        sample_rate: Audio sample rate
        segment_seconds: Segment length (should match training)
        overlap: Overlap fraction between chunks
        device: Device to use (auto-detect if None)
    """
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model = load_model(checkpoint_path, device)

    # Create dataset for the single song
    segment_samples = int(sample_rate * segment_seconds)
    dataset = MusDBStemDataset(
        root_dir=data_dir,
        segment_samples=segment_samples,
        sample_rate=sample_rate,
        random_segments=False,
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Main inference loop
    filename = dataset.files[0]
    all_stems = dataset._load_stems(filename)
    all_stems = torch.from_numpy(all_stems).permute(0, 2, 1).float()  # (num_stems, channels, samples)
    all_stems = all_stems.to(device)
    full_mixure = all_stems[0]   # (C, T)
    length = all_stems.shape[-1]                              # Full track T
    final = torch.zeros((len(STEMS), 2, all_stems.shape[-1]), device=device)     # NOTE: Assuming stereo output
    for stem_idx, stem_name in enumerate(STEMS):
        # ============================================================
        # Compute chunking parameters
        # ============================================================
        chunk_len = int(sample_rate * segment_seconds)      # check if (1 + overlap) is needed here
        overlap_frames = int(overlap * sample_rate)

        fade = Fade(
            fade_in_len=0,
            fade_out_len=overlap_frames,
            fade_shape="linear"
        )

        start = 0
        end = chunk_len

        print(f"Extracting stem '{stem_name}'...")

        # ============================================================
        # Chunked streaming loop for given `stem_name`
        # ============================================================
        while start < length:
            end = min(start + chunk_len, length)

            chunk = full_mixure[:, start:end].unsqueeze(0).to(device)   # (1, C, T_chunk)
            print(f"start: {start}, end: {end}")

            with torch.no_grad():
                out = model(chunk, stem_name)    # (B, 1, C, T_chunk); Also using stem name as prompt (this might be an issue for 'other' stem)

            # Determine fades for this chunk
            fade_in = 0 if start == 0 else overlap_frames
            fade_out = overlap_frames if end < length else 0

            fade = Fade(
                fade_in_len=fade_in,
                fade_out_len=fade_out,
                fade_shape="linear"
            )

            out = fade(out)

            # Flatten batch
            out = out.squeeze(0)   # -> (C, T_chunk)

            # Add into final buffer
            final[stem_idx, :, start:end] += out

            # Move window
            start += (chunk_len - overlap_frames)


    # ============================================================
    # Compute SDR for each stem (after chunked processing)
    # ============================================================
    sdr_scores = {stem: -30.0 for stem in STEMS}
    
    for i in range(min(len(STEMS), 4)):  # Limit to true 4 stems for SDR computation
        stem_name = STEMS[i]

        estimate = final[i, :, :]   # (C, T)
        sdr = -sdr_loss(estimate, all_stems[i + 1]).item()          # Stem 0 is mixture
        sdr_scores[stem_name] = sdr
        print(f"{stem_name:8s} | SDR: {sdr:6.2f} dB")

    # Save FULL stem tracks to output directory
    if output_dir:
        cleaned_filename = Path(filename).stem.replace(".stem", "")
        cleaned_filename = cleaned_filename.replace("-", "")
        cleaned_filename = cleaned_filename.replace("'", "")
        cleaned_filename = cleaned_filename.replace(" ", "_")
        full_dir = Path(output_dir) / Path(cleaned_filename)
        for i in range(len(STEMS)):
            stem_name = STEMS[i]
            estimate = final[i, :, :]
            Path(full_dir).mkdir(parents=True, exist_ok=True)
            audio_np = estimate.cpu().numpy().T                             # (T, C)
            output_file = full_dir / f"extracted_{stem_name}.wav"
            sf.write(str(output_file), audio_np, sample_rate)

        # Also save mixture for reference
        mixture_np = full_mixure.cpu().numpy().T                             # (T, C)
        output_file = full_dir / f"mixture.wav"
        sf.write(str(output_file), mixture_np, sample_rate)
        
    # TODO: Plot spectrograms of original vs extracted stems for full track length
    for i in range(len(STEMS)):
        stem_name = STEMS[i]
        estimate = final[i, :, :]
        fig = plot_separation_spectrograms(full_mixure, estimate, all_stems[i + 1], stem_name, sample_rate)
        fig.show()
    
    # Wait for user to close all plots
    plt.show()

    # # Summary
    # print("\n" + "=" * 60)
    # print("SDR Summary (averaged over segments):")
    # print("=" * 60)

    # all_sdrs = []
    # for stem in STEMS:
    #     if sdr_scores[stem]:
    #         avg_sdr = sum(sdr_scores[stem]) / len(sdr_scores[stem])
    #         all_sdrs.append(avg_sdr)
    #         print(f"  {stem:8s}: {avg_sdr:6.2f} dB (n={len(sdr_scores[stem])})")

    # if all_sdrs:
    #     print(f"  {'Average':8s}: {sum(all_sdrs) / len(all_sdrs):6.2f} dB")

    # print("=" * 60)
    # print(f"Results saved to: {output_path}")

    return sdr_scores


if __name__ == "__main__":
    test_inference(
        checkpoint_path="checkpoints/2025_12_04/best_model.pt",
        data_dir="/home/jacob/datasets/musdb18/inference2",
        output_dir="results/2025_12_05",
        device='cpu',
    )