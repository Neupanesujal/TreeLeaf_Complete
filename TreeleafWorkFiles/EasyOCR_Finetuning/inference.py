import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from model import get_model
from utils import decode_prediction
from config import MODEL_PATH, IMAGE_SIZE, DEVICE
from dataset import LicensePlateDataset
from config import *

def load_model(num_classes):
    model = get_model(num_classes)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE)
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).to(DEVICE)

def infer(model, image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output, 2)
    return predicted.squeeze().cpu().numpy()

def decode_prediction(prediction, idx_to_char):
    return ''.join([idx_to_char[idx] for idx in prediction if idx != 0])

def display_result(image_path, prediction):
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Predicted: {prediction}")
    plt.show()

if __name__ == "__main__":
    dataset = LicensePlateDataset(TRAIN_DIR, TRAIN_LABELS)
    num_classes = dataset.get_num_classes()
    idx_to_char = dataset.idx_to_char

    model = load_model(num_classes)
    image_path = "./dataset/en_train_filtered/5.png"
    prediction = infer(model, image_path)
    print("here is error ",prediction)
    decoded_prediction = decode_prediction(prediction, idx_to_char)
    print(f"Predicted license plate: {decoded_prediction}")
    display_result(image_path, decoded_prediction)