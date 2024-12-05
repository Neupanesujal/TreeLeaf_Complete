import torch
import easyocr


model = easyocr.Reader(['ne'])

# Path to your image
image_path = 'dataset/en_train_filtered/1.JPG'

# Perform OCR
results = model.readtext(image_path)

# Print the extracted text
for (bbox, text, confidence) in results:
    print(f"Detected text: {text} (Confidence: {confidence:.2f})")
