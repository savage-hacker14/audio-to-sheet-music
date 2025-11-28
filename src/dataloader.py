import random
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset
import stempeg
import numpy as np

# ============================================================================
# Data Loader
# ============================================================================

def get_random_prompt(stem_name: str) -> str:
    """Get a random text prompt for a given stem."""
    return random.choice(STEM_PROMPTS[stem_name])


# Text Prompt Templates
STEM_PROMPTS: Dict[str, List[str]] = {
    "drums": ["drums", "drum kit", "percussion", "the drums"],
    "bass": ["bass", "bass guitar", "the bass", "bass line"],
    "other": ["other instruments", "accompaniment", "instruments"],
    "vocals": ["vocals", "voice", "singing", "the vocals"],
}

STEM_NAME_TO_INDEX = {"drums": 0, "bass": 1, "other": 2, "vocals": 3}


class MusDBStemDataset(Dataset):
    def __init__(
            self,
            root_dir: str,
            segment_samples: int,
            sample_rate: int = 44100,
            channels: int = 2,
            random_segments: bool = True,
            augment: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.segment_samples = segment_samples
        self.sample_rate = sample_rate
        self.channels = channels
        self.random_segments = random_segments
        self.augment = augment

        self.files = list(self.root_dir.glob("*.stem.mp4"))
        if not self.files:
            raise ValueError(f"No .stem.mp4 files found in {root_dir}")

        print(f"Found {len(self.files)} tracks in {root_dir}")
        self.stem_names = ["drums", "bass", "other", "vocals"]

    def __len__(self) -> int:
        return len(self.files) * len(self.stem_names)

    def _load_stems(self, filepath: Path) -> np.ndarray:
        """Load all stems from a .stem.mp4 file."""
        stems, rate = stempeg.read_stems(str(filepath))
        # stems shape: (num_stems, samples, channels)
        # [mix, drums, bass, other, vocals]
        return stems

    def _extract_segment(self, stems: np.ndarray) -> np.ndarray:
        """Extract the same random segment from all stems."""
        total_samples = stems.shape[1]  # stems: (num_stems, samples, channels)

        if total_samples <= self.segment_samples:
            # Pad if too short
            pad_amount = self.segment_samples - total_samples
            stems = np.pad(stems, ((0, 0), (0, pad_amount), (0, 0)), mode='constant')
        else:
            # Random start position (same for all stems)
            if self.random_segments:
                start = random.randint(0, total_samples - self.segment_samples)
            else:
                start = 0
            stems = stems[:, start:start + self.segment_samples, :]

        return stems

    def _augment(self, mixture: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation."""
        if random.random() < 0.5:
            gain = random.uniform(0.7, 1.3)
            mixture = mixture * gain
            target = target * gain

        if random.random() < 0.3 and mixture.shape[-1] == 2:
            mixture = mixture[:, ::-1].copy()
            target = target[:, ::-1].copy()

        return mixture, target

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        file_idx = idx // len(self.stem_names)
        stem_idx = idx % len(self.stem_names)

        filepath = self.files[file_idx]
        stem_name = self.stem_names[stem_idx]

        # Load all stems with resampling handled by stempeg
        stems = self._load_stems(filepath)

        # Extract same segment from all stems
        stems = self._extract_segment(stems)

        # stems: [mix, drums, bass, other, vocals]
        mixture = stems[0]  # (T, C)
        target = stems[stem_idx + 1]  # (T, C)

        # Augmentation
        if self.augment:
            mixture, target = self._augment(mixture, target)

        # Convert to tensors: (T, C) -> (C, T)
        mixture_tensor = torch.from_numpy(mixture.T).float()
        target_tensor = torch.from_numpy(target.T).float()

        # Ensure stereo
        if mixture_tensor.shape[0] == 1:
            mixture_tensor = mixture_tensor.repeat(2, 1)
            target_tensor = target_tensor.repeat(2, 1)

        prompt = get_random_prompt(stem_name)

        return {
            "mixture": mixture_tensor,
            "target": target_tensor,
            "prompt": prompt,
            "stem_name": stem_name,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor | List[str]]:
    """Custom collate function."""
    return {
        "mixture": torch.stack([item["mixture"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "prompt": [item["prompt"] for item in batch],
        "stem_name": [item["stem_name"] for item in batch],
    }