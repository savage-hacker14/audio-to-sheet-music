import random
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset
import stempeg
import soundfile as sf
import math
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

        self.stem_names = ["drums", "bass", "other", "vocals"]

        self.files = list(self.root_dir.glob("*.stem.mp4"))
        if not self.files:
            raise ValueError(f"No .stem.mp4 files found in {root_dir}")
        
        # Compute number of examples
        self.index_map      = []                        # (file_idx, stem_idx, segment_idx)
        #self.sample_lengths = [0] * len(self.files)     # total samples per file
        for file_idx, file in enumerate(self.files):
            info = stempeg.Info(str(file))
            total_samples = info.duration(0) * info.sample_rate(0)      # 0 - using mixture stream as reference
            #self.sample_lengths[file_idx] = int(total_samples)
            num_segments = math.ceil(total_samples / segment_samples)

            # Build index map: for each stem, each segment
            for stem_idx in range(len(self.stem_names)):
                for seg in range(num_segments):
                    self.index_map.append((file_idx, stem_idx, seg))

        print(f"Found {len(self.files)} tracks, total dataset items: {len(self.index_map)}")

    def __len__(self) -> int:
        return len(self.index_map)

    def _load_stems(self, filepath: Path) -> np.ndarray:
        """Load all stems from a .stem.mp4 file."""
        stems, rate = stempeg.read_stems(str(filepath))
        # stems shape: (num_stems, samples, channels)
        # [mix, drums, bass, other, vocals]
        return stems

    def _extract_random_segment(self, stems: np.ndarray) -> np.ndarray:
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
    
    def _extract_segment(self, stems: np.ndarray, seg_idx: int) -> np.ndarray:
        total_samples = stems.shape[1]

        if self.random_segments:
            # fallback to random segment extractor
            return self._extract_random_segment(stems)

        start = seg_idx * self.segment_samples
        end = start + self.segment_samples

        if end <= total_samples:
            return stems[:, start:end, :]
        else:
            # Last segment may need padding
            pad_amount = end - total_samples
            seg = stems[:, start:, :]
            seg = np.pad(seg, ((0, 0),(0, pad_amount), (0, 0)), mode="constant")
            return seg

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
        file_idx, stem_idx, seg_idx = self.index_map[idx]

        filepath = self.files[file_idx]
        stems = self._load_stems(filepath)

        # deterministic segment selection
        stems = self._extract_segment(stems, seg_idx)

        mixture = stems[0]               # (T, C)
        target  = stems[stem_idx+1]      # (T, C)

        if self.augment:
            mixture, target = self._augment(mixture, target)

        # -> (C, T)
        mixture = torch.from_numpy(mixture.T).float()
        target  = torch.from_numpy(target.T).float()

        # ensure stereo
        if mixture.shape[0] == 1:
            mixture = mixture.repeat(2, 1)
            target  = target.repeat(2, 1)

        prompt = get_random_prompt(self.stem_names[stem_idx])

        return {
            "mixture": mixture,
            "target": target,
            "prompt": prompt,
            "stem_name": self.stem_names[stem_idx],
            "file_idx": file_idx,
            "segment_idx": seg_idx,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor | List[str]]:
    """Custom collate function."""
    return {
        "mixture": torch.stack([item["mixture"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "prompt": [item["prompt"] for item in batch],
        "stem_name": [item["stem_name"] for item in batch],
    }