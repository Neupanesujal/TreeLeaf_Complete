import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import easyocr
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Configuration Constants
CONFIG = {
    
    'TRAIN_IMAGE_DIR': 'dataset/en_train_filtered',
    'TRAIN_CSV_PATH': 'dataset/en_train_filtered/labels.csv',
    'EVAL_IMAGE_DIR': 'dataset/en_val',
    'EVAL_CSV_PATH': 'dataset/en_val/labels.csv',
    
    
    'MODEL_SAVE_PATH': 'nepali_ocr_model',
    'PLOT_SAVE_PATH': 'training_metrics.png',
    
    
    'LEARNING_RATE': 1e-4,
    'BATCH_SIZE': 16,
    'NUM_EPOCHS': 50,
    
    
    'INPUT_HEIGHT': 224,
    'INPUT_WIDTH': 224,
    
    
    'LANGUAGE': ['ne'],  
}

class CharacterMapper:
    def __init__(self, labels):
        
        all_chars = set(''.join(labels))
        self.char_to_index = {char: idx+1 for idx, char in enumerate(sorted(all_chars))}
        self.char_to_index['<blank>'] = 0 
        self.index_to_char = {idx: char for char, idx in self.char_to_index.items()}
        self.num_classes = len(self.char_to_index)
    
    def encode(self, text):
        return torch.tensor([self.char_to_index.get(char, 0) for char in text], dtype=torch.long)
    
    def decode(self, indices):
        return ''.join([self.index_to_char.get(idx, '') for idx in indices])

class NepaliLicensePlateDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        
        
        self.char_mapper = CharacterMapper(self.data['words'].tolist())
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        img_path = os.path.join(self.image_dir, self.data.iloc[idx]['filename'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        if self.transform:
            image = self.transform(image)
        
        
        label = self.data.iloc[idx]['words']
        encoded_label = self.char_mapper.encode(label)
        
        return image, encoded_label, len(encoded_label)

def custom_collate(batch):
    
    batch.sort(key=lambda x: x[2], reverse=True)
    
    
    images, labels, label_lengths = zip(*batch)
    
    
    images = torch.stack(images, 0)
    
    
    max_len = max(label_lengths)
    padded_labels = torch.zeros(len(labels), max_len, dtype=torch.long)
    for i, label in enumerate(labels):
        padded_labels[i, :len(label)] = label
    
    
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)
    
    return images, padded_labels, label_lengths

class OCRTrainer:
    def __init__(self, language, config):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
       
        self.reader = easyocr.Reader(language, gpu=torch.cuda.is_available())
        
       
        self.config = config
        
       
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config['INPUT_HEIGHT'], config['INPUT_WIDTH'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
       
        train_dataset = NepaliLicensePlateDataset(
            config['TRAIN_CSV_PATH'], 
            config['TRAIN_IMAGE_DIR'], 
            self.transform
        )
        
        eval_dataset = NepaliLicensePlateDataset(
            config['EVAL_CSV_PATH'], 
            config['EVAL_IMAGE_DIR'], 
            self.transform
        )
        
        
        self.char_mapper = train_dataset.char_mapper
        
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=config['BATCH_SIZE'], 
            shuffle=True,
            collate_fn=custom_collate
        )
        
        self.eval_loader = DataLoader(
            eval_dataset, 
            batch_size=config['BATCH_SIZE'], 
            shuffle=False,
            collate_fn=custom_collate
        )
        
       
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True).to(self.device)

    def train(self):
        
        train_losses, eval_losses = [], []
        train_accuracies, eval_accuracies = [], []
        
        
        for epoch in range(self.config['NUM_EPOCHS']):
            
            train_loss, train_accuracy = self._run_epoch(self.train_loader, is_training=True)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            
            
            eval_loss, eval_accuracy = self._run_epoch(self.eval_loader, is_training=False)
            eval_losses.append(eval_loss)
            eval_accuracies.append(eval_accuracy)
            
            
            print(f"Epoch {epoch+1}: "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Train Accuracy: {train_accuracy:.4f}, "
                  f"Eval Loss: {eval_loss:.4f}, "
                  f"Eval Accuracy: {eval_accuracy:.4f}")
        
        
        self._plot_metrics(train_losses, eval_losses, train_accuracies, eval_accuracies)
    
    def _run_epoch(self, dataloader, is_training=True):
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for images, labels, label_lengths in dataloader:
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            label_lengths = label_lengths.to(self.device)
            
            
            batch_size = images.size(0)
            
            
            log_probs = []
            input_lengths = []
            
            for img, label, label_len in zip(images, labels, label_lengths):
                
                img_np = img.cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np * 255).astype(np.uint8)
                
                
                result = self.reader.readtext(img_np)
                
                
                if result:
                    pred_text = result[0][1]
                    pred_label = self.char_mapper.encode(pred_text)
                    
                    
                    log_prob = torch.log_softmax(
                        torch.rand(pred_label.size(0), self.char_mapper.num_classes, device=self.device), 
                        dim=1
                    )
                    
                    log_probs.append(log_prob)
                    input_lengths.append(log_prob.size(0))
                else:
                   
                    log_prob = torch.log_softmax(
                        torch.rand(1, self.char_mapper.num_classes, device=self.device), 
                        dim=1
                    )
                    log_probs.append(log_prob)
                    input_lengths.append(1)
            
            
            max_input_len = max(input_lengths)
            padded_log_probs = torch.zeros(
                batch_size, max_input_len, self.char_mapper.num_classes, 
                device=self.device
            )
            
            for i, log_prob in enumerate(log_probs):
                padded_log_probs[i, :log_prob.size(0), :] = log_prob
            
            
            log_probs_ctc = padded_log_probs.transpose(0, 1)
            
            
            input_lengths = torch.tensor(input_lengths, device=self.device)
            
            
            loss = self.ctc_loss(
                log_probs_ctc, 
                labels, 
                input_lengths, 
                label_lengths
            )
            
            total_loss += loss.item()
            total_samples += batch_size
            
            
            for i in range(batch_size):
                pred_text = self.reader.readtext(
                    (images[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                
                if pred_text:
                    pred_text = pred_text[0][1]
                    true_text = self.char_mapper.decode(labels[i][:label_lengths[i]].cpu().numpy())
                    
                    if pred_text == true_text:
                        total_correct += 1
        
        
        epoch_loss = total_loss / len(dataloader)
        epoch_accuracy = total_correct / total_samples
        
        return epoch_loss, epoch_accuracy
    
    def _plot_metrics(self, train_losses, eval_losses, train_accuracies, eval_accuracies):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(eval_losses, label='Eval Loss')
        plt.title('Training and Evaluation Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(eval_accuracies, label='Eval Accuracy')
        plt.title('Training and Evaluation Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.config['PLOT_SAVE_PATH'])
        plt.close()

def main():
    ocr_trainer = OCRTrainer(CONFIG['LANGUAGE'], CONFIG)
    ocr_trainer.train()

if __name__ == '__main__':
    main()