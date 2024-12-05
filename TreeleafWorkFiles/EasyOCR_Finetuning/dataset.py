# dataset.py

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_SIZE = (320, 100)  # (width, height)

class LicensePlateDataset(Dataset):
    def __init__(self, data_dir, csv_file):
        self.data_dir = data_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.char_to_idx = {char: idx+1 for idx, char in enumerate(set(''.join(self.data['words'])))}
        self.char_to_idx['<PAD>'] = 0  # Add padding token
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.max_label_length = max(len(label) for label in self.data['words'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.data.iloc[idx, 1]
        
        image = self.transform(image)
        
        # Encode and pad the label
        label_encoded = [self.char_to_idx[c] for c in label]
        label_length = len(label_encoded)
        label_encoded = label_encoded + [0] * (self.max_label_length - label_length)
        label_encoded = torch.tensor(label_encoded, dtype=torch.long)
        
        # Create a mask for the actual label length
        mask = torch.zeros(self.max_label_length, dtype=torch.bool)
        mask[:label_length] = 1
        
        return image, label_encoded, mask, label_length

    def get_num_classes(self):
        return len(self.char_to_idx)

    def get_max_label_length(self):
        return self.max_label_length