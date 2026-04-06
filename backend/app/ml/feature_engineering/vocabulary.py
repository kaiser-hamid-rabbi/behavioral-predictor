"""Vocabulary loader and builder for categorical features."""

from __future__ import annotations

import json
from pathlib import Path


class Vocabulary:
    """Manages mappings between string categories and integer embeddings."""
    
    def __init__(self) -> None:
        self.mappings: dict[str, dict[str, int]] = {
            "event_name": {"<PAD>": 0, "<UNK>": 1},
            "device_os": {"<PAD>": 0, "<UNK>": 1},
            "channel": {"<PAD>": 0, "<UNK>": 1},
            "traffic_source": {"<PAD>": 0, "<UNK>": 1},
            "category": {"<PAD>": 0, "<UNK>": 1},
        }

    def build_from_dataframe(self, df) -> None:
        """
        Builds vocabularies dynamically from a pandas DataFrame of events.
        """
        for col in self.mappings.keys():
            if col in df.columns:
                unique_vals = df[col].dropna().unique().tolist()
                for val in unique_vals:
                    if str(val) not in self.mappings[col]:
                        self.mappings[col][str(val)] = len(self.mappings[col])

    def encode(self, feature_name: str, value: str | None) -> int:
        """Convert a categorical string to its integer ID."""
        val_str = str(value) if value is not None else "<UNK>"
        return self.mappings.get(feature_name, {}).get(val_str, 1)  # 1 is <UNK>

    def get_vocab_size(self, feature_name: str) -> int:
        """Get the size of a specific vocabulary."""
        return len(self.mappings.get(feature_name, []))

    def save(self, output_dir: str | Path) -> None:
        """Save vocabularies to JSON files for browser use."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        for feature_name, mapping in self.mappings.items():
            with open(path / f"{feature_name}.json", "w") as f:
                json.dump(mapping, f, indent=2)

    def load(self, input_dir: str | Path) -> None:
        """Load vocabularies from JSON files."""
        path = Path(input_dir)
        for feature_name in self.mappings.keys():
            file_path = path / f"{feature_name}.json"
            if file_path.exists():
                with open(file_path, "r") as f:
                    self.mappings[feature_name] = json.load(f)
