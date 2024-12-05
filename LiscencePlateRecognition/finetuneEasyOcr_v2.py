import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50


CONFIG = {
    
    'TRAIN_IMAGE_DIR': 'dataset/en_train_filtered',
    'TRAIN_CSV_PATH': 'dataset/en_train_filtered/labels.csv',
    'EVAL_IMAGE_DIR': 'dataset/en_val',
    'EVAL_CSV_PATH': 'dataset/en_val/labels.csv',
    
    
    'MODEL_SAVE_PATH': 'nepali_ocr_model.pth',
    'PLOT_SAVE_PATH': 'training_metrics.png',
    
    
    'LEARNING_RATE': 1e-4,
    'BATCH_SIZE': 16,
    'NUM_EPOCHS': 50,
    
   
    'INPUT_HEIGHT': 224,
    'INPUT_WIDTH': 224,
}

class CharacterMapper:
    def __init__(self, labels):
        
        all_chars = set(''.join(labels))
        self.char_to_index = {char: idx for idx, char in enumerate(sorted(all_chars))}
        self.index_to_char = {idx: char for char, idx in self.char_to_index.items()}
        self.num_classes = len(self.char_to_index)
    
    def encode(self, text):
        return torch.tensor([self.char_to_index.get(char, -1) for char in text], dtype=torch.long)
    
    def decode(self, indices):
        return ''.join([self.index_to_char.get(idx, '') for idx in indices if idx != -1])

def custom_collate(batch):
    
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    

    images, labels = zip(*batch)
    

    images = torch.stack(images, 0)
    

    max_len = len(labels[0])
    padded_labels = torch.full((len(labels), max_len), -1, dtype=torch.long)
    
    for i, label in enumerate(labels):
        padded_labels[i, :len(label)] = label
    
    return images, padded_labels

class NepaliLicensePlateDataset(Dataset):
    def __init__(self, csv_path, image_dir, char_mapper, transform=None):
 
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.char_mapper = char_mapper
    
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
        
        return image, encoded_label

class NepaliOCRModel(nn.Module):
    def __init__(self, num_classes):
        super(NepaliOCRModel, self).__init__()
        
       
        backbone = resnet50(pretrained=True)
        
       
        for param in backbone.parameters():
            param.requires_grad = False
        
   
        for param in backbone.layer4.parameters():
            param.requires_grad = True
        

        self.features = nn.Sequential(*list(backbone.children())[:-1])
        

        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        
        features = self.features(x)
        features = features.view(features.size(0), -1)
        
      
        outputs = self.classifier(features)
        
        return outputs

class OCRTrainer:
    def __init__(self, config):
       
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
       
        self.config = config
        
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config['INPUT_HEIGHT'], config['INPUT_WIDTH'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        
        train_data = pd.read_csv(config['TRAIN_CSV_PATH'])
        self.char_mapper = CharacterMapper(train_data['words'].tolist())
        
     
        train_dataset = NepaliLicensePlateDataset(
            config['TRAIN_CSV_PATH'], 
            config['TRAIN_IMAGE_DIR'], 
            self.char_mapper,
            self.transform
        )
        
        eval_dataset = NepaliLicensePlateDataset(
            config['EVAL_CSV_PATH'], 
            config['EVAL_IMAGE_DIR'], 
            self.char_mapper,
            self.transform
        )
        
        
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
        
      
        self.model = NepaliOCRModel(self.char_mapper.num_classes).to(self.device)
        
        
        self.optimizer = optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad], 
            lr=config['LEARNING_RATE']
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

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
        
        
        torch.save(self.model.state_dict(), self.config['MODEL_SAVE_PATH'])
        
      
        self._plot_metrics(train_losses, eval_losses, train_accuracies, eval_accuracies)

    def _run_epoch(self, dataloader, is_training=True):
      
        self.model.train() if is_training else self.model.eval()
        
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
       
        with torch.set_grad_enabled(is_training):
            for images, labels in dataloader:
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
               
                if is_training:
                    self.optimizer.zero_grad()
                
                
                outputs = self.model(images)
                
                
                loss = 0
                batch_correct = 0
                batch_tokens = 0
                
                for i in range(labels.size(0)):
                    
                    valid_mask = labels[i] != -1
                    valid_labels = labels[i][valid_mask]
                    
                    if len(valid_labels) > 0:
                        
                        sample_outputs = outputs[i].unsqueeze(0).repeat(len(valid_labels), 1)
                        
                        
                        sample_loss = self.criterion(sample_outputs, valid_labels)
                        loss += sample_loss
                        
                       
                        predictions = torch.argmax(sample_outputs, dim=1)
                        batch_correct += (predictions == valid_labels).float().sum().item()
                        batch_tokens += len(valid_labels)
                
         
                loss = loss / labels.size(0)
                
                
                if is_training:
                    loss.backward()
                    self.optimizer.step()
                
                
                total_loss += loss.item()
                total_correct += batch_correct
                total_tokens += batch_tokens
        
       
        epoch_loss = total_loss / len(dataloader)
        epoch_accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        
        return epoch_loss, epoch_accuracy
    
    def _plot_metrics(self, train_losses, eval_losses, train_accuracies, eval_accuracies):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(eval_losses, label='Eval Loss')
        plt.title('Training and Evaluation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(eval_accuracies, label='Eval Accuracy')
        plt.title('Training and Evaluation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.config['PLOT_SAVE_PATH'])
        plt.close()

def main():
    
    ocr_trainer = OCRTrainer(CONFIG)
    ocr_trainer.train()

if __name__ == '__main__':
    main()