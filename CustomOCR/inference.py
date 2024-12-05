import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import joblib

class OCRInference:
    def __init__(self, model_path='custom_ocr_model', encoder_path='label_encoder.pkl'):
        # Load the trained model
        self.model = load_model(model_path)
        
        # Load the label encoder
        self.label_encoder = joblib.load(encoder_path)
    
    def preprocess_image(self, image_path):
        # Load and preprocess the image
        img = load_img(image_path, target_size=(32, 320), color_mode='grayscale')
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    def predict(self, image_path):
        # Preprocess the image
        processed_image = self.preprocess_image(image_path)
        
        # Make prediction
        prediction = self.model.predict(processed_image)
        
        # Get the predicted label
        pred_class = np.argmax(prediction, axis=1)
        pred_label = self.label_encoder.inverse_transform(pred_class)
        
        return pred_label[0]

# Example usage
if __name__ == '__main__':
    # Path to the image you want to predict
    image_path = 'dataset/en_val/604.JPG'
    
    # Create inference object
    ocr_inferencer = OCRInference()
    
    # Predict the text in the image
    predicted_text = ocr_inferencer.predict(image_path)
    print(f"Predicted Text: {predicted_text}")