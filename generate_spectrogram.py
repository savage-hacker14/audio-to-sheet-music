"""
Generate spectrograms for one or more tracks.

Usage:
    python generate_spectrogram.py "Tom McKenzie - Directions"
    python generate_spectrogram.py --top5
    python generate_spectrogram.py "Track 1" "Track 2" "Track 3" --wandb
"""

import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from benchmark import OurModel, load_track_stems, STEMS, SAMPLE_RATE
from utils import plot_all_stems_spectrograms, plot_separation_spectrograms, log_spectrogram_to_wandb


# Top 5 tracks by average SDR
TOP5_TRACKS = [
    "Speak Softly - Like Horses",
    "Tom McKenzie - Directions",
    "BKS - Too Much",
    "Lyndsey Ollard - Catching Up",
    "The Mountaineering Club - Mallory",
]


def process_track(track_name, model, test_dir, output_dir, use_wandb):
    """Process a single track and generate spectrograms."""
    
    # Find the track file
    track_file = test_dir / f"{track_name}.stem.mp4"
    if not track_file.exists():
        print(f"Track not found: {track_file}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Processing: {track_name}")
    print('='*60)
    
    # Load track
    print("Loading track...")
    mixture, reference_stems = load_track_stems(track_file)
    
    # Separate all stems
    print("Separating stems...")
    estimated_stems = model.separate_all(mixture)
    
    # Ensure same length for all
    min_len = min(
        mixture.shape[-1],
        min(est.shape[-1] for est in estimated_stems.values()),
        min(ref.shape[-1] for ref in reference_stems.values())
    )
    
    mixture = mixture[:, :min_len]
    estimated_stems = {k: v[:, :min_len].cpu() for k, v in estimated_stems.items()}
    reference_stems = {k: v[:, :min_len] for k, v in reference_stems.items()}
    
    # Generate combined plot (all stems)
    print("Generating spectrograms...")
    fig = plot_all_stems_spectrograms(
        mixture=mixture,
        estimated_stems=estimated_stems,
        reference_stems=reference_stems,
        sample_rate=SAMPLE_RATE,
        figsize=(16, 14),
    )
    
    # Save combined plot
    safe_name = track_name.replace(" ", "_").replace("'", "")
    output_path = output_dir / f"{safe_name}_all_stems.png"
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    # Log to W&B
    if use_wandb:
        import wandb
        wandb.log({f"spectrograms/{safe_name}/all_stems": wandb.Image(fig, caption=f"{track_name} - All Stems")})
    plt.close(fig)
    
    # Generate individual stem plots
    for stem in STEMS:
        fig = plot_separation_spectrograms(
            mixture=mixture,
            estimated=estimated_stems[stem],
            reference=reference_stems[stem],
            stem_name=stem,
            sample_rate=SAMPLE_RATE,
        )
        
        output_path = output_dir / f"{safe_name}_{stem}.png"
        fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
        
        # Log to W&B
        if use_wandb:
            import wandb
            wandb.log({f"spectrograms/{safe_name}/{stem}": wandb.Image(fig, caption=f"{track_name} - {stem.capitalize()}")})
        plt.close(fig)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate spectrograms for one or more tracks")
    parser.add_argument("track_names", type=str, nargs="*",
                        help="Track name(s) without .stem.mp4")
    parser.add_argument("--top5", action="store_true", 
                        help="Process all top 5 tracks by SDR")
    parser.add_argument("--wandb", action="store_true", help="Log spectrograms to W&B")
    parser.add_argument("--wandb-project", type=str, default="audio-text-htdemucs-spectrograms",
                        help="W&B project name")
    args = parser.parse_args()
    
    # Determine which tracks to process
    if args.top5:
        track_names = TOP5_TRACKS
    elif args.track_names:
        track_names = args.track_names
    else:
        track_names = ["Tom McKenzie - Directions"]  # Default
    
    print(f"Will process {len(track_names)} track(s):")
    for t in track_names:
        print(f"  - {t}")
    
    # Paths
    test_dir = Path("../musdb18/test")
    output_dir = Path("results/spectrograms")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize W&B if requested
    if args.wandb:
        import wandb
        run_name = "top5-spectrograms" if args.top5 else f"spectrogram-{len(track_names)}-tracks"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={"tracks": track_names}
        )
        print(f"W&B initialized: {args.wandb_project}")
    
    # Set device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # Load model (once, reuse for all tracks)
    checkpoint = "/Users/surya/Downloads/2025_12_01_batch4/best_model.pt"
    print("Loading model...")
    model = OurModel(checkpoint, device)
    
    # Process each track
    successful = 0
    for track_name in track_names:
        if process_track(track_name, model, test_dir, output_dir, args.wandb):
            successful += 1
    
    # Finish W&B run
    if args.wandb:
        import wandb
        wandb.finish()
        print("W&B run finished.")
    
    print(f"\n{'='*60}")
    print(f"Done! Processed {successful}/{len(track_names)} tracks")
    print(f"Spectrograms saved to: {output_dir}")
    print('='*60)


if __name__ == "__main__":
    main()

