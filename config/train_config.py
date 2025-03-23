from dataclasses import dataclass
import torch

@dataclass
class TrainConfig:
    lr: float = 3e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 25
