from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as transforms
import torch

# Load the YOLOv8 model
model = YOLO('best.pt')  # Assuming 'best.pt' is your trained YOLOv8 model

# Define the image preprocessing steps
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to the expected input size
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize with 3 channels (RGB)
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Ensure the image has 3 channels (RGB)
    image = transform(image)  # Apply transformations
    image = image.unsqueeze(0)  # Add a batch dimension (1, 3, H, W)
    return image

def decode_output(output):
    # Assuming output is a tensor, and this is a typical OCR model output
    predicted_text = "".join([chr(torch.argmax(char_prob)) for char_prob in output[0]])
    return predicted_text

def clean_text(text):
    cleaned_text = text.strip()
    return cleaned_text

# Inference process
image_path = 'Capture.jpeg'
input_image = preprocess_image(image_path)

# Perform inference with the YOLO model
with torch.no_grad():
    results = model(input_image)  # YOLO model inference

# The output format might differ depending on how the model was trained
# Assuming you need to decode the text from the model output:
output = results[0].probs  # Adjust this based on your specific model output format
text_output = decode_output(output)
final_text = clean_text(text_output)

print("Final Extracted Text:", final_text)
