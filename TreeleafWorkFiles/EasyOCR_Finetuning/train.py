# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import LicensePlateDataset
from model import get_model
from utils import plot_metrics, calculate_metrics
from config import *

def custom_collate(batch):
    images, labels, masks, lengths = zip(*batch)
    images = torch.stack(images, 0)
    labels = torch.stack(labels, 0)
    masks = torch.stack(masks, 0)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return images, labels, masks, lengths

def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    
    train_dataset = LicensePlateDataset(TRAIN_DIR, TRAIN_LABELS)
    val_dataset = LicensePlateDataset(VAL_DIR, VAL_LABELS)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=custom_collate)

    
    num_classes = train_dataset.get_num_classes()
    model = get_model(num_classes).to(DEVICE)

   
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.1)

    
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels, masks, lengths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            images, labels, masks, lengths = images.to(DEVICE), labels.to(DEVICE), masks.to(DEVICE), lengths.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            
            output_lengths = torch.full(size=(outputs.size(0),), fill_value=outputs.size(1), dtype=torch.long).to(DEVICE)
            loss = criterion(outputs.log_softmax(2).transpose(0, 1), labels, output_lengths, lengths)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            
            _, predicted = outputs.max(2)
            for i, length in enumerate(lengths):
                train_correct += (predicted[i, :length] == labels[i, :length]).sum().item()
                train_total += length.item()

        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

      
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels, masks, lengths in val_loader:
                images, labels, masks, lengths = images.to(DEVICE), labels.to(DEVICE), masks.to(DEVICE), lengths.to(DEVICE)
                outputs = model(images)
                
                output_lengths = torch.full(size=(outputs.size(0),), fill_value=outputs.size(1), dtype=torch.long).to(DEVICE)
                loss = criterion(outputs.log_softmax(2).transpose(0, 1), labels, output_lengths, lengths)

                val_loss += loss.item()
                
                
                _, predicted = outputs.max(2)
                for i, length in enumerate(lengths):
                    val_correct += (predicted[i, :length] == labels[i, :length]).sum().item()
                    val_total += length.item()

        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)

   
    torch.save(model.state_dict(), MODEL_PATH)

  
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

    
    calculate_metrics(model, val_loader, num_classes)

if __name__ == "__main__":
    train()