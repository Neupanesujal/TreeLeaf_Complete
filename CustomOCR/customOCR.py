import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import save_model

# I disabled GPU to avoid CUDA errors. 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class CustomOCRTrainer:
    def __init__(self, image_dir, labels_path):
        self.image_dir = image_dir
        self.labels_df = pd.read_csv(labels_path)
        
    def prepare_dataset(self):
        images = []
        labels = []
        
        for _, row in self.labels_df.iterrows():
            img_path = os.path.join(self.image_dir, row['filename'])
            
            try:
                img = load_img(img_path, target_size=(32, 320), color_mode='grayscale')
                img_array = img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(row['words'])
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        return np.array(images), labels
    
    def create_ocr_model(self, input_shape, num_classes):
        inputs = Input(shape=input_shape)
        x = Conv2D(32, (3, 3), activation='relu')(inputs)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def train_custom_recognizer(self, test_size=0.2):
        
        X, y = self.prepare_dataset()
        
        
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        y_categorical = to_categorical(y_encoded)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=test_size, random_state=42
        )
        

        model = self.create_ocr_model(input_shape=(32, 320, 1), num_classes=y_categorical.shape[1])
        history = model.fit(X_train, y_train, 
                  validation_data=(X_test, y_test),
                  epochs=20, batch_size=32)
        
        return model, label_encoder, history
    
    def plot_training_history(self, history):

        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')


        plt.subplot(1,2,2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()

    def evaluate_ocr(self, model, label_encoder):
        X, y = self.prepare_dataset()
        y_encoded = label_encoder.transform(y)
        y_categorical = to_categorical(y_encoded)
        
        predictions = model.predict(X)
        pred_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
        
        results = pd.DataFrame({
            'filename': self.labels_df['filename'],
            'ground_truth': y,
            'prediction': pred_labels
        })
        
        return results


trainer = CustomOCRTrainer(
    image_dir='dataset/en_train_filtered', 
    labels_path='dataset/en_train_filtered/labels.csv'
)


custom_model, label_encoder, training_history = trainer.train_custom_recognizer()


trainer.plot_training_history(training_history)


save_model(custom_model, 'custom_ocr_model')


import joblib
joblib.dump(label_encoder, 'label_encoder.pkl')


evaluation_results = trainer.evaluate_ocr(custom_model, label_encoder)
print(evaluation_results)