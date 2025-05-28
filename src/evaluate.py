import os
import sys
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import EmotionRecognitionModel
from src.model_2 import ImprovedEmotionRecognitionModel
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

def evaluate_model(data_dir='data', model_path=None, batch_size=64, model_version='original'):
    if model_path is None:
        raise ValueError("You must specify the path to the model checkpoint (best_model.pth).")
    checkpoint_folder = os.path.dirname(model_path)
    _, val_loader = get_dataloaders(data_dir, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    if model_version == 'improved':
        print("Using improved emotion recognition model")
        model = ImprovedEmotionRecognitionModel(num_classes=7).to(device)
    else:
        print("Using original emotion recognition model")
        model = EmotionRecognitionModel(num_classes=7).to(device)
    
    # Load checkpoint - handle both simple state_dict or full checkpoint format
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # This is a full checkpoint with training state
        print(f"Loading from full checkpoint (epoch {checkpoint['epoch']})")
        model.load_state_dict(checkpoint['model_state_dict'])
        # Extract training metrics for reporting
        best_val_loss = checkpoint.get('best_val_loss', 'N/A')
        print(f"Best validation loss from checkpoint: {best_val_loss}")
    else:
        # Legacy format - direct state_dict
        print("Loading from legacy model format (state_dict only)")
        model.load_state_dict(checkpoint)
    
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate performance metrics
    report = classification_report(all_labels, all_preds, target_names=val_loader.dataset.classes, output_dict=True)
    report_text = classification_report(all_labels, all_preds, target_names=val_loader.dataset.classes)
    
    # Display comprehensive results
    print("Classification Report:")
    print(report_text)
    
    # Add overall metrics summary
    accuracy = report['accuracy']
    macro_avg = report['macro avg']['f1-score']
    weighted_avg = report['weighted avg']['f1-score']
    print(f"\nOverall Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Avg F1: {macro_avg:.4f}")
    print(f"Weighted Avg F1: {weighted_avg:.4f}")
    
    # Add checkpoint info if available
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        epoch = checkpoint.get('epoch', 'N/A')
        best_val_loss = checkpoint.get('best_val_loss', 'N/A')
        print(f"\nCheckpoint Info:")
        print(f"Epoch: {epoch}")
        print(f"Best Val Loss: {best_val_loss:.4f}" if isinstance(best_val_loss, float) else f"Best Val Loss: {best_val_loss}")

    # Save classification report to txt file in checkpoint folder
    report_path = os.path.join(checkpoint_folder, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("Classification Report:\n")
        f.write(report_text)
        f.write(f"\nOverall Metrics:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro Avg F1: {macro_avg:.4f}\n")
        f.write(f"Weighted Avg F1: {weighted_avg:.4f}\n")
        
        # Add checkpoint info to the report file
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            epoch = checkpoint.get('epoch', 'N/A')
            best_val_loss = checkpoint.get('best_val_loss', 'N/A')
            f.write(f"\nCheckpoint Info:\n")
            f.write(f"Epoch: {epoch}\n")
            f.write(f"Best Val Loss: {best_val_loss:.4f}\n" if isinstance(best_val_loss, float) else f"Best Val Loss: {best_val_loss}\n")

    # Generate and plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create a more informative plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=val_loader.dataset.classes, 
                yticklabels=val_loader.dataset.classes, ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    
    # Plot normalized percentages
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens', 
                xticklabels=val_loader.dataset.classes,
                yticklabels=val_loader.dataset.classes, ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized)')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    
    # Add overall title with model info
    model_info = f"Model: {model_version}"
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        epoch = checkpoint.get('epoch', 'N/A')
        model_info += f" (Epoch {epoch})"
    
    plt.suptitle(f"{model_info} - Accuracy: {accuracy:.4f}", fontsize=16)
    plt.tight_layout()
    
    # Autosave confusion matrix figure in checkpoint folder
    fig_path = os.path.join(checkpoint_folder, 'confusion_matrix.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--model_path', type=str, required=True, help='Path to best_model.pth')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_version', type=str, default='original', choices=['original', 'improved'], 
                        help='Select model architecture (original or improved)')
    args = parser.parse_args()
    evaluate_model(
        data_dir=args.data_dir,
        model_path=args.model_path,
        batch_size=args.batch_size,
        model_version=args.model_version
    )
