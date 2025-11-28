from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from demucs import pretrained
from transformers import AutoTokenizer, ClapModel

from src.models.stem_separation.ATHTDemucs_v2 import AudioTextHTDemucs
from src.loss import combined_loss, sdr_loss
from src.dataloader import MusDBStemDataset, collate_fn, STEM_PROMPTS


# ============================================================================
# Training Helper Functions
# ============================================================================

def train_epoch(
        model: AudioTextHTDemucs,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[GradScaler],
        device: str,
        use_amp: bool,
        grad_clip: float,
        sdr_weight: float,
        sisdr_weight: float,
        epoch: int,
        log_every: int,
        use_wandb: bool,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_sdr = 0.0
    total_sisdr = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

    for batch_idx, batch in enumerate(pbar):
        mixture = batch["mixture"].to(device)
        target = batch["target"].to(device)
        prompts = batch["prompt"]

        optimizer.zero_grad()

        if use_amp and device == "cuda":
            with autocast():
                estimated = model(mixture, prompts)
                loss, metrics = combined_loss(
                    estimated, target, sdr_weight, sisdr_weight
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            estimated = model(mixture, prompts)
            loss, metrics = combined_loss(
                estimated, target, sdr_weight, sisdr_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += metrics["loss/total"]
        total_sdr += metrics["metrics/sdr"]
        total_sisdr += metrics["metrics/sisdr"]
        num_batches += 1

        pbar.set_postfix({
            "loss": f"{metrics['loss/total']:.4f}",
            "SDR": f"{metrics['metrics/sdr']:.2f}",
        })

        if use_wandb and batch_idx % log_every == 0:
            import wandb
            wandb.log({
                "train/loss": metrics["loss/total"],
                "train/sdr": metrics["metrics/sdr"],
                "train/sisdr": metrics["metrics/sisdr"],
                "train/step": epoch * len(dataloader) + batch_idx,
            })

    return {
        "loss": total_loss / num_batches,
        "sdr": total_sdr / num_batches,
        "sisdr": total_sisdr / num_batches,
    }


@torch.no_grad()
def validate(
        model: AudioTextHTDemucs,
        dataloader: DataLoader,
        device: str,
        use_amp: bool,
        sdr_weight: float = 0.9,
        sisdr_weight: float = 0.1,
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()

    total_loss = 0.0
    total_sdr = 0.0
    total_sisdr = 0.0
    num_batches = 0

    stem_metrics = {name: {"sdr": 0.0, "count": 0} for name in STEM_PROMPTS.keys()}

    for batch in tqdm(dataloader, desc="Validating"):
        mixture = batch["mixture"].to(device)
        target = batch["target"].to(device)
        prompts = batch["prompt"]
        stem_names = batch["stem_name"]

        if use_amp and device == "cuda":
            with autocast():
                estimated = model(mixture, prompts)
                loss, metrics = combined_loss(estimated, target, sdr_weight, sisdr_weight)
        else:
            estimated = model(mixture, prompts)
            loss, metrics = combined_loss(estimated, target, sdr_weight, sisdr_weight)

        total_loss += metrics["loss/total"]
        total_sdr += metrics["metrics/sdr"]
        total_sisdr += metrics["metrics/sisdr"]
        num_batches += 1

        for i, stem_name in enumerate(stem_names):
            est_i = estimated[i:i + 1]
            tgt_i = target[i:i + 1]
            sdr_i = -sdr_loss(est_i, tgt_i).item()
            stem_metrics[stem_name]["sdr"] += sdr_i
            stem_metrics[stem_name]["count"] += 1

    avg_metrics = {
        "loss": total_loss / num_batches,
        "sdr": total_sdr / num_batches,
        "sisdr": total_sisdr / num_batches,
    }

    for stem_name, data in stem_metrics.items():
        if data["count"] > 0:
            avg_metrics[f"sdr/{stem_name}"] = data["sdr"] / data["count"]

    return avg_metrics


def save_checkpoint(
        model: AudioTextHTDemucs,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        epoch: int,
        metrics: Dict[str, float],
        checkpoint_dir: str,
        is_best: bool = False,
):
    """Save a training checkpoint."""
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
    }

    path = checkpoint_path / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")

    if is_best:
        best_path = checkpoint_path / "best_model.pt"
        torch.save(checkpoint, best_path)
        print(f"Saved best model to {best_path}")

    latest_path = checkpoint_path / "latest.pt"
    torch.save(checkpoint, latest_path)


def load_checkpoint(
        model: AudioTextHTDemucs,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        checkpoint_path: str,
) -> int:
    """Load a checkpoint and return the epoch number."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return checkpoint["epoch"]


# ============================================================================
# Main Training Function
# ============================================================================

def train(
        # Paths
        train_dir: str = "../data/train",
        test_dir: str = "../data/test",
        checkpoint_dir: str = "../checkpoints",

        # Audio parameters
        sample_rate: int = 44100,
        segment_seconds: float = 6.0,

        # Training parameters
        batch_size: int = 4,
        num_workers: int = 4,
        epochs: int = 10,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        grad_clip: float = 5.0,

        # Loss weights
        sdr_weight: float = 0.9,
        sisdr_weight: float = 0.1,

        # Model parameters
        model_dim: int = 384,
        text_dim: int = 512,
        n_heads: int = 8,

        # Logging
        use_wandb: bool = False,
        wandb_project: str = "audio-text-htdemucs",
        wandb_run_name: Optional[str] = None,
        log_every: int = 50,
        validate_every: int = 1,
        save_every: int = 5,

        # Mixed precision
        use_amp: bool = True,

        # Device
        device: Optional[str] = None,

        # Resume training
        resume_from: Optional[str] = None,
):
    """
    Main training function for AudioTextHTDemucs.

    Args:
        train_dir: Path to training data directory
        test_dir: Path to test/validation data directory
        checkpoint_dir: Path to save checkpoints
        sample_rate: Audio sample rate
        segment_seconds: Length of audio segments in seconds
        batch_size: Training batch size
        num_workers: Number of dataloader workers
        epochs: Number of training epochs
        learning_rate: Initial learning rate
        weight_decay: AdamW weight decay
        grad_clip: Gradient clipping value
        sdr_weight: Weight for SDR loss component
        sisdr_weight: Weight for SI-SDR loss component
        model_dim: Model hidden dimension
        text_dim: Text embedding dimension
        n_heads: Number of attention heads
        use_wandb: Whether to use Weights & Biases logging
        wandb_project: W&B project name
        wandb_run_name: W&B run name (optional)
        log_every: Log training metrics every N batches
        validate_every: Run validation every N epochs
        save_every: Save checkpoint every N epochs
        use_amp: Use automatic mixed precision
        device: Device to train on (auto-detected if None)
        resume_from: Path to checkpoint to resume from (optional)

    Returns:
        Dict containing final metrics and best SDR achieved
    """
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    segment_samples = int(sample_rate * segment_seconds)

    # Initialize wandb
    if use_wandb:
        import wandb
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "train_dir": train_dir,
                "test_dir": test_dir,
                "sample_rate": sample_rate,
                "segment_seconds": segment_seconds,
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "grad_clip": grad_clip,
                "sdr_weight": sdr_weight,
                "sisdr_weight": sisdr_weight,
                "model_dim": model_dim,
                "text_dim": text_dim,
                "n_heads": n_heads,
                "use_amp": use_amp,
            },
        )

    print("=" * 60)
    print("Audio-Text HTDemucs Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Train directory: {train_dir}")
    print(f"Test directory: {test_dir}")
    print(f"Segment length: {segment_seconds}s ({segment_samples} samples)")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 60)

    # Load pretrained models
    print("Loading pretrained HTDemucs...")
    htdemucs = pretrained.get_model('htdemucs').models[0]

    print("Loading CLAP model...")
    clap = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

    # Create model
    print("Building AudioTextHTDemucs model...")
    model = AudioTextHTDemucs(
        htdemucs_model=htdemucs,
        clap_encoder=clap,
        clap_tokenizer=tokenizer,
        model_dim=model_dim,
        text_dim=text_dim,
        num_heads=n_heads,
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create datasets
    print("Creating datasets...")
    train_dataset = MusDBStemDataset(
        root_dir=train_dir,
        segment_samples=segment_samples,
        sample_rate=sample_rate,
        random_segments=True,
        augment=True,
    )

    val_dataset = MusDBStemDataset(
        root_dir=test_dir,
        segment_samples=segment_samples,
        sample_rate=sample_rate,
        random_segments=False,
        augment=False,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
    )

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=learning_rate * 0.01,
    )

    # Mixed precision scaler
    scaler = GradScaler() if use_amp and device == "cuda" else None

    # Resume from checkpoint
    start_epoch = 0
    best_sdr = -float("inf")

    if resume_from is not None:
        resume_path = Path(resume_from)
        if resume_path.exists():
            print(f"Resuming from {resume_path}")
            start_epoch = load_checkpoint(model, optimizer, scheduler, str(resume_path))
            start_epoch += 1
    else:
        # Check for latest checkpoint
        latest_checkpoint = Path(checkpoint_dir) / "latest.pt"
        if latest_checkpoint.exists():
            print(f"Found latest checkpoint at {latest_checkpoint}")
            start_epoch = load_checkpoint(model, optimizer, scheduler, str(latest_checkpoint))
            start_epoch += 1

    # Training loop
    print("\nStarting training...")
    for epoch in range(start_epoch, epochs):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.2e}")
        print(f"{'=' * 60}")

        # Train
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            grad_clip=grad_clip,
            sdr_weight=sdr_weight,
            sisdr_weight=sisdr_weight,
            epoch=epoch,
            log_every=log_every,
            use_wandb=use_wandb,
        )
        print(f"Train - Loss: {train_metrics['loss']:.4f}, SDR: {train_metrics['sdr']:.2f} dB")

        # Step scheduler
        scheduler.step()

        # Validate
        if (epoch + 1) % validate_every == 0:
            val_metrics = validate(
                model=model,
                dataloader=val_loader,
                device=device,
                use_amp=use_amp,
                sdr_weight=sdr_weight,
                sisdr_weight=sisdr_weight,
            )
            print(f"Val - Loss: {val_metrics['loss']:.4f}, SDR: {val_metrics['sdr']:.2f} dB")

            for stem_name in STEM_PROMPTS.keys():
                if f"sdr/{stem_name}" in val_metrics:
                    print(f"  {stem_name}: {val_metrics[f'sdr/{stem_name}']:.2f} dB")

            if use_wandb:
                import wandb
                wandb.log({
                    "val/loss": val_metrics["loss"],
                    "val/sdr": val_metrics["sdr"],
                    "val/sisdr": val_metrics["sisdr"],
                    **{f"val/{k}": v for k, v in val_metrics.items() if k.startswith("sdr/")},
                    "epoch": epoch + 1,
                })

            is_best = val_metrics["sdr"] > best_sdr
            if is_best:
                best_sdr = val_metrics["sdr"]
                print(f"New best SDR: {best_sdr:.2f} dB")
        else:
            val_metrics = {}
            is_best = False

        # Save checkpoint
        if (epoch + 1) % save_every == 0 or is_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1,
                {**train_metrics, **val_metrics},
                checkpoint_dir, is_best
            )
        else:
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1,
                {**train_metrics, **val_metrics},
                checkpoint_dir, is_best=False
            )

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation SDR: {best_sdr:.2f} dB")
    print("=" * 60)

    if use_wandb:
        import wandb
        wandb.finish()

    return {
        "final_train_metrics": train_metrics,
        "final_val_metrics": val_metrics,
        "best_sdr": best_sdr,
    }


if __name__ == "__main__":
    # Example: run training with default parameters
    train()