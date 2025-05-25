import os
import sys
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import EmotionRecognitionModel
from src.data_preprocessing import get_dataloaders
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob

def get_latest_checkpoint(checkpoints_dir='checkpoints'):
    folders = glob.glob(os.path.join(checkpoints_dir, 'model_*'))
    if not folders:
        raise FileNotFoundError("No checkpoint folders found.")
    latest_folder = max(folders, key=os.path.getmtime)
    model_path = os.path.join(latest_folder, 'best_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model file found in {latest_folder}")
    print(f"Loading model from checkpoint folder: {latest_folder}")
    return model_path

def evaluate_model(data_dir='data', checkpoints_dir='checkpoints', batch_size=64):
    _, val_loader = get_dataloaders(data_dir, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = EmotionRecognitionModel(num_classes=7).to(device)
    model_path = get_latest_checkpoint(checkpoints_dir)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=val_loader.dataset.classes))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=val_loader.dataset.classes, yticklabels=val_loader.dataset.classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    evaluate_model()
