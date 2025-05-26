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


def get_checkpoint_from_dir(checkpoint_dir='./checkpoints/model_20250525_201104/'):
    model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model file found in {checkpoint_dir}")
    print(f"Loading model from checkpoint directory: {checkpoint_dir}")
    return model_path, checkpoint_dir

def evaluate_model(data_dir='data', model_path=None, batch_size=64):
    if model_path is None:
        raise ValueError("You must specify the path to the model checkpoint (best_model.pth).")
    checkpoint_folder = os.path.dirname(model_path)
    _, val_loader = get_dataloaders(data_dir, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = EmotionRecognitionModel(num_classes=7).to(device)
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

    report = classification_report(all_labels, all_preds, target_names=val_loader.dataset.classes)
    print("Classification Report:")
    print(report)

    # Save classification report to txt file in checkpoint folder
    report_path = os.path.join(checkpoint_folder, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("Classification Report:\n")
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=val_loader.dataset.classes, yticklabels=val_loader.dataset.classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    # Autosave confusion matrix figure in checkpoint folder
    fig_path = os.path.join(checkpoint_folder, 'confusion_matrix.png')
    plt.savefig(fig_path)
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--model_path', type=str, required=True, help='Path to best_model.pth')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    evaluate_model(data_dir=args.data_dir, model_path=args.model_path, batch_size=args.batch_size)
