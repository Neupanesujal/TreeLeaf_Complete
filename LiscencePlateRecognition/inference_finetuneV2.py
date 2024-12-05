import os
import torch
import cv2
import numpy as np
from torchvision import transforms
import pandas as pd


CONFIG = {
    
    'TRAIN_IMAGE_DIR': 'dataset/en_train_filtered',
    'TRAIN_CSV_PATH': 'dataset/en_train_filtered/labels.csv',
    'EVAL_IMAGE_DIR': 'dataset/en_val',
    'EVAL_CSV_PATH': 'dataset/en_val/labels.csv',
    'MODEL_SAVE_PATH': 'nepali_ocr_model.pth',
    'PLOT_SAVE_PATH': 'training_metrics.png',
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
        if not isinstance(indices, list):
            indices = [indices]
        return ''.join([self.index_to_char.get(idx, '') for idx in indices if idx != -1])

class NepaliOCRModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(NepaliOCRModel, self).__init__()
        
        
        backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        
        
        for param in backbone.parameters():
            param.requires_grad = False
        
        
        for param in backbone.layer4.parameters():
            param.requires_grad = True
        
        
        self.features = torch.nn.Sequential(*list(backbone.children())[:-1])
        
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        
        features = self.features(x)
        features = features.view(features.size(0), -1)
        
       
        outputs = self.classifier(features)
        
        return outputs

def load_and_preprocess_image(image_path):
    
    image = cv2.imread(image_path)
    
    
    if image is None:
        raise ValueError(f"Unable to load image from {image_path}")
    
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((CONFIG['INPUT_HEIGHT'], CONFIG['INPUT_WIDTH'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
   
    transformed_image = transform(image)
    
    return transformed_image

def load_model(model_path, num_classes):
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    

    model = NepaliOCRModel(num_classes).to(device)
    

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
   
    model.eval()
    
    return model, device

def inference(model, image, char_mapper, device):

    image = image.unsqueeze(0).to(device)
    
   
    with torch.no_grad():
    
        output = model(image)
        
        
        predictions = torch.argmax(output, dim=1)
        
      
        predicted_chars = char_mapper.decode(predictions.cpu().numpy()[0])
    
    return predicted_chars

def main():
   
    model_path = CONFIG['MODEL_SAVE_PATH']
    
    
    try:
        train_data = pd.read_csv(CONFIG['TRAIN_CSV_PATH'])
    except FileNotFoundError:
        print(f"Training CSV not found at {CONFIG['TRAIN_CSV_PATH']}")
        return
    
    char_mapper = CharacterMapper(train_data['words'].tolist())
    
  
    try:
        model, device = load_model(model_path, char_mapper.num_classes)
    except Exception as e:
        print(f"Model loading failed: {e}")
        return
    
 
    image_path = 'dataset/en_train_filtered/1.png'
    
   
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    try:
        
        processed_image = load_and_preprocess_image(image_path)
        prediction = inference(model, processed_image, char_mapper, device)
        print("Predicted Text:", prediction)
    
    except Exception as e:
        print(f"Inference failed: {e}")

if __name__ == '__main__':
    main()