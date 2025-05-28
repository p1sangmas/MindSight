import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import sys
import os
from datetime import datetime
from tqdm import tqdm
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import EmotionRecognitionModel
from src.model_2 import ImprovedEmotionRecognitionModel, FocalLoss
from src.data_preprocessing import get_dataloaders
from sklearn.metrics import accuracy_score
import numpy as np
from collections import Counter


def train_model(data_dir='data', save_dir='checkpoints', log_dir='runs', num_epochs=30, batch_size=64, lr=1e-3, patience=5, early_stopping=True, model_folder=None, oversample=False, model_version='original', use_focal_loss=False, class_weights=None, resume_from=None):
    # Use specified folder name if provided, else use timestamp
    if model_folder is not None:
        run_save_dir = os.path.join(save_dir, model_folder)
        run_log_dir = os.path.join(log_dir, f'run_{model_folder}')
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_log_dir = os.path.join(log_dir, f'run_{timestamp}')
        run_save_dir = os.path.join(save_dir, f'model_{timestamp}')
    os.makedirs(run_save_dir, exist_ok=True)
    writer = SummaryWriter(run_log_dir)

    train_loader, val_loader = get_dataloaders(data_dir, batch_size, oversample=oversample)

    # Count class distribution in training set
    train_labels_list = []
    for _, labels in train_loader:
        train_labels_list.extend(labels.cpu().numpy().tolist())
    class_counts = Counter(train_labels_list)
    total_samples = len(train_labels_list)
    print(f"Total training samples: {total_samples}")
    print("Class distribution:")
    for cls, count in sorted(class_counts.items()):
        print(f"  Class {cls}: {count}")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training on {len(train_loader.dataset)} training samples and {len(val_loader.dataset)} validation samples.")
    
    # Select model based on model_version parameter
    if model_version == 'improved':
        print("Using improved emotion recognition model")
        model = ImprovedEmotionRecognitionModel(num_classes=7).to(device)
    else:  # default to original
        print("Using original emotion recognition model")
        model = EmotionRecognitionModel(num_classes=7).to(device)
    
    # Load weights if resuming from a checkpoint
    if resume_from:
        print(f"Loading checkpoint from: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_val_loss = checkpoint['best_val_loss']
        start_epoch = checkpoint['epoch'] + 1
        epochs_no_improve = checkpoint['epochs_no_improve']
        print(f"Resuming from epoch {start_epoch} with best val loss {best_val_loss:.4f}")
    
    # Select loss function based on parameters
    if use_focal_loss:
        print("Using Focal Loss")
        criterion = FocalLoss(gamma=2)
    elif class_weights is not None:
        print(f"Using weighted CrossEntropyLoss with weights: {class_weights}")
        weights = torch.tensor(class_weights, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loss, train_preds, train_labels = 0.0, [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_preds.extend(outputs.argmax(1).detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': loss.item()})

        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(train_labels, train_preds)

        model.eval()
        val_loss, val_preds, val_labels = 0.0, [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_preds.extend(outputs.argmax(1).detach().cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(val_labels, val_preds)

        elapsed = time.time() - start_time
        log_line = (f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Time: {elapsed:.2f}s\n")
        print(log_line.strip())
        writer.add_scalars('Loss', {'Train': train_loss, 'Val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'Train': train_acc, 'Val': val_acc}, epoch)
        writer.add_scalar('Time/Epoch', elapsed, epoch)

        # Save log to file
        log_file_path = os.path.join(run_save_dir, 'train_log.txt')
        with open(log_file_path, 'a') as f:
            f.write(log_line)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'epochs_no_improve': epochs_no_improve
            }
            torch.save(checkpoint, os.path.join(run_save_dir, 'best_model.pth'))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if early_stopping and epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    writer.close()
    print("Training complete.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='runs')
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--model_folder', type=str, default=None, help='Specify folder name for this run (inside checkpoints/)')
    parser.add_argument('--oversample', action='store_true', help='Enable class balancing with oversampling')
    parser.add_argument('--model_version', type=str, default='original', choices=['original', 'improved'], help='Select model architecture')
    parser.add_argument('--focal_loss', action='store_true', help='Use Focal Loss instead of CrossEntropyLoss')
    parser.add_argument('--class_weights', type=float, nargs='+', default=None, help='Class weights for loss function [angry, disgust, fear, happy, neutral, sad, surprise]')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to model checkpoint to resume training from')
    args = parser.parse_args()
    train_model(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        early_stopping=args.early_stopping,
        model_folder=args.model_folder,
        oversample=args.oversample,
        model_version=args.model_version,
        use_focal_loss=args.focal_loss,
        class_weights=args.class_weights,
        resume_from=args.resume_from
    )
