import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet

from config.model_config import ModelConfig

config = ModelConfig()

transform = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = OxfordIIITPet(root="./notebooks", split="trainval", download=False, transform=transform)

#============
# Dataloader
# ===========
train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)

sample_images, sample_labels = next(iter(train_loader))

print("Batch shape:", sample_images.shape)  # (B, C, H, W)
print("Labels shape:", sample_labels.shape)  # (B,)

import matplotlib.pyplot as plt

def show_images(images, labels, num=8):
    plt.figure(figsize=(12, 6))
    for i in range(num):
        img = images[i].permute(1, 2, 0).numpy()  # Convert (C, H, W) -> (H, W, C)
        img = (img * 0.5) + 0.5  # Unnormalize
        plt.subplot(2, 4, i + 1)
        plt.imshow(img)
        plt.title(f"Label: {labels[i].item()}")
        plt.axis("off")
    plt.show()

show_images(sample_images, sample_labels)
