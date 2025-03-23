import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from config.model_config import ModelConfig
from model.model import ViT
from model.dataloader import train_loader
from config.train_config import TrainConfig

def train():
    model_config = ModelConfig()
    train_config = TrainConfig()

    # model
    model = ViT(model_config).to(train_config.device)

    # Set up optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=train_config.lr)
    loss_fn = nn.CrossEntropyLoss()

    # Create directories for checkpoints and logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"vit_run_{timestamp}"
    
    os.makedirs("checkpoints", exist_ok=True)
    writer = SummaryWriter(f"logs/{run_name}")

    # Training loop
    model.train()
    best_loss = float('inf')
    
    for epoch in range(train_config.epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_config.epochs}", leave=True)

        for batch_idx, (images, labels) in enumerate(loop):
            images, labels = images.to(train_config.device), labels.to(train_config.device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            loop.set_postfix(
                loss=loss.item(),
                acc=100.*correct/total,
                avg_loss=running_loss/(batch_idx+1)
            )
        
        # Epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        # Log to tensorboard
        writer.add_scalar('training/loss', epoch_loss, epoch)
        writer.add_scalar('training/accuracy', epoch_acc, epoch)
        
        # Save model checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, f"checkpoints/{run_name}_best.pth")
    
    # Save final model
    torch.save({
        'epoch': train_config.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
    }, f"checkpoints/{run_name}_final.pth")
    
    print(f"Training completed. Best loss: {best_loss:.4f}")
    
    return model

if __name__ == "__main__":
    train()