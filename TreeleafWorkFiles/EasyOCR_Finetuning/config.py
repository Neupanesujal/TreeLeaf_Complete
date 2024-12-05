import os
import torch

BASE_DIR = "./"
DATA_DIR = "./dataset"
TRAIN_DIR = './dataset/en_train_filtered/'
VAL_DIR = './dataset/en_val/'
TRAIN_LABELS = './dataset/en_train_filtered/labels.csv'
VAL_LABELS = './dataset/en_val/labels.csv'


CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
MODEL_PATH = os.path.join(BASE_DIR, 'nepali_license_plate_model.pth')

IMAGE_SIZE = (320, 100)  # (width, height)
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
CHECKPOINT_INTERVAL = 4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')