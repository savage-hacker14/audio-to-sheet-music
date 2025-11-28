import torch
import soundfile as sf
from pathlib import Path

from demucs import pretrained
from transformers import ClapModel, AutoTokenizer
from torch.utils.data import DataLoader

from src.models.stem_separation.ATHTDemucs_v2 import AudioTextHTDemucs
from src.dataloader import MusDBStemDataset, collate_fn, STEM_PROMPTS
from src.loss import sdr_loss


STEMS = ["drums", "bass", "other", "vocals"]

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
    model.load_state_dict(checkpoint["model_state_dict"])
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
        device: str = None,
):
    """
    Run inference on a single song, compute SDR for all stems, save results.

    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Directory containing a single .stem.mp4 file
        output_dir: Directory to save separated stems
        sample_rate: Audio sample rate
        segment_seconds: Segment length (should match training)
        device: Device to use (auto-detect if None)
    """
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

    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, num_workers=0)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning inference on {len(dataset)} samples...")
    print("=" * 60)

    sdr_scores = {stem: [] for stem in STEMS}

    for batch in loader:
        mixture = batch["mixture"].to(device)
        target = batch["target"].to(device)
        prompt = batch["prompt"][0]
        stem_name = batch["stem_name"][0]

        # Run inference
        estimated = model(mixture, batch["prompt"])

        # Compute SDR
        sdr = -sdr_loss(estimated, target).item()
        sdr_scores[stem_name].append(sdr)

        print(f"{stem_name:8s} | Prompt: '{prompt:20s}' | SDR: {sdr:6.2f} dB")

        # Save the estimated stem
        output_file = output_path / f"{stem_name}_{prompt.replace(' ', '_')}.wav"
        audio_np = estimated.squeeze(0).cpu().numpy().T  # (C, T) -> (T, C)
        sf.write(str(output_file), audio_np, sample_rate)

    # Summary
    print("\n" + "=" * 60)
    print("SDR Summary (averaged over segments):")
    print("=" * 60)

    all_sdrs = []
    for stem in STEMS:
        if sdr_scores[stem]:
            avg_sdr = sum(sdr_scores[stem]) / len(sdr_scores[stem])
            all_sdrs.append(avg_sdr)
            print(f"  {stem:8s}: {avg_sdr:6.2f} dB (n={len(sdr_scores[stem])})")

    if all_sdrs:
        print(f"  {'Average':8s}: {sum(all_sdrs) / len(all_sdrs):6.2f} dB")

    print("=" * 60)
    print(f"Results saved to: {output_path}")

    return sdr_scores


if __name__ == "__main__":
    test_inference(
        checkpoint_path="checkpoints/best_model.pt",
        data_dir="data/quick_train",
        output_dir="results",
        device=None,
    )