from typing import Dict, Tuple
import torch


# ============================================================================
# Loss Functions
# ============================================================================

def sdr_loss(estimated, target):
    """
    Compute negative SDR loss.
    Based on the definition from Vincent et al. 2006.
    """
    # Flatten to [batch, -1] to ensure compatible shapes
    est_flat = estimated.reshape(estimated.shape[0], -1)
    tgt_flat = target.reshape(target.shape[0], -1)

    # Compute SDR: 10 * log10(||target||^2 / ||target - estimated||^2)
    delta = 1e-8  # Small constant for numerical stability

    num = torch.sum(tgt_flat ** 2, dim=-1)
    den = torch.sum((tgt_flat - est_flat) ** 2, dim=-1)

    # Avoid division by zero
    sdr = 10 * torch.log10((num + delta) / (den + delta))

    # Clamp to reasonable range to avoid extreme values
    sdr = torch.clamp(sdr, min=-30, max=30)

    return -sdr.mean()  # Return negative for minimization


def sisdr_loss(estimated, target):
    """
    Compute negative SI-SDR (Scale-Invariant SDR) loss.
    This is more robust to scaling differences between estimate and target.
    """
    # Flatten to [batch, -1]
    est_flat = estimated.reshape(estimated.shape[0], -1)
    tgt_flat = target.reshape(target.shape[0], -1)

    # Zero-mean normalization (critical for SI-SDR)
    est_flat = est_flat - est_flat.mean(dim=-1, keepdim=True)
    tgt_flat = tgt_flat - tgt_flat.mean(dim=-1, keepdim=True)

    # SI-SDR calculation
    # Project estimate onto target: s_target = <s', s> / ||s||^2 * s
    delta = 1e-8

    dot = torch.sum(est_flat * tgt_flat, dim=-1, keepdim=True)
    s_target_norm_sq = torch.sum(tgt_flat ** 2, dim=-1, keepdim=True)

    # Scaled target
    s_target = (dot / (s_target_norm_sq + delta)) * tgt_flat

    # Noise is the orthogonal component
    e_noise = est_flat - s_target

    # SI-SDR = 10 * log10(||s_target||^2 / ||e_noise||^2)
    s_target_energy = torch.sum(s_target ** 2, dim=-1)
    e_noise_energy = torch.sum(e_noise ** 2, dim=-1)

    sisdr = 10 * torch.log10((s_target_energy + delta) / (e_noise_energy + delta))

    # Clamp to reasonable range
    sisdr = torch.clamp(sisdr, min=-30, max=30)

    return -sisdr.mean()  # Return negative for minimization


def new_sdr_metric(estimated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the SDR according to the MDX challenge definition (positive values).
    This is for evaluation/logging, not for loss.

    Args:
        estimated: (batch, channels, time)
        target: (batch, channels, time)

    Returns:
        SDR scores per batch item (batch,)
    """
    delta = 1e-8
    num = torch.sum(target ** 2, dim=(1, 2))
    den = torch.sum((target - estimated) ** 2, dim=(1, 2))
    scores = 10 * torch.log10((num + delta) / (den + delta))
    return scores


def combined_loss(
        estimated: torch.Tensor,
        target: torch.Tensor,
        sdr_weight: float = 0.9,
        sisdr_weight: float = 0.1
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combined SDR and SI-SDR loss.

    Args:
        estimated: Estimated audio (batch, channels, time)
        target: Target audio (batch, channels, time)
        sdr_weight: Weight for SDR loss (default 0.9)
        sisdr_weight: Weight for SI-SDR loss (default 0.1)

    Returns:
        total_loss: Combined loss for backpropagation
        metrics: Dictionary of metrics for logging
    """
    sdr = sdr_loss(estimated, target)
    sisdr = sisdr_loss(estimated, target)

    total = sdr_weight * sdr + sisdr_weight * sisdr

    # For logging, also compute positive SDR metric
    with torch.no_grad():
        pos_sdr = new_sdr_metric(estimated, target).mean()

    metrics = {
        "loss/total": total.item(),
        "loss/sdr": sdr.item(),
        "loss/sisdr": sisdr.item(),
        "metrics/sdr": -sdr.item(),  # Positive SDR for logging
        "metrics/sisdr": -sisdr.item(),  # Positive SI-SDR for logging
        "metrics/new_sdr": pos_sdr.item(),  # MDX-style SDR
    }

    return total, metrics


def combined_L1_sdr_loss(
        estimated: torch.Tensor,
        target: torch.Tensor,
        sdr_weight: float = 1.0,
        l1_weight: float = 0.05
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combined SDR and L1 loss.

    Args:
        estimated: Estimated audio (batch, channels, time)
        target: Target audio (batch, channels, time)
        sdr_weight: Weight for SDR loss (default 0.9)
        l1_weight: Weight for SI-SDR loss (default 0.1)
    Returns:
        total_loss: Combined loss for backpropagation
        metrics: Dictionary of metrics for logging
    """
    sdr = sdr_loss(estimated, target)
    sisdr = sisdr_loss(estimated, target)
    l1  = torch.nn.functional.l1_loss(estimated, target)

    total = sdr_weight * sdr + l1_weight * l1
    
    metrics = {
        "loss/total": total.item(),
        "loss/sdr": sdr.item(),
        "loss/sisdr": sisdr.item(),
        "metrics/sdr": -sdr.item(),  # Positive SDR for logging
        "metrics/sisdr": -sisdr.item(),  # Positive SI-SDR for logging
    }

    return total, metrics