import cv2
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model

def preprocess_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize the image to the size expected by your model
    resized_image = cv2.resize(image, (32, 32))

    # Normalize the image data (if required by your model)
    #normalized_image = resized_image / 255.0

    # Expand dimensions to match the model input shape (e.g., (1, 128, 128, 1))
    input_image = np.expand_dims(resized_image, axis=[0, -1])

    return input_image
     

def decode_prediction(pred):
    # This function should convert the model's output into a readable text format
    # Replace this with your model's decoding logic
    # For instance, mapping indexes to characters, applying a CTC decoder, etc.
    return ''.join([chr(np.argmax(x)) for x in pred])

def clean_text(text):
    # Remove unwanted characters, spaces, or apply any other cleaning steps
    cleaned_text = text.strip()
    return cleaned_text

# Complete process

# 1. Load the model
model = load_model('model.h5')

# 2. Preprocess the image
image_path = 'Capture.jpeg'
processed_image = preprocess_image(image_path)

# 3. Perform inference
predicted_text = model.predict(processed_image)

# 4. Decode and clean the text output
text_output = decode_prediction(predicted_text)
final_text = clean_text(text_output)

print("Final Extracted Text:", text_output)
print("Final Extracted Text:", final_text)

