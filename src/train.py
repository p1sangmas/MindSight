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
from src.data_preprocessing import get_dataloaders
from sklearn.metrics import accuracy_score
import numpy as np


def train_model(data_dir='data', save_dir='checkpoints', log_dir='runs', num_epochs=30, batch_size=64, lr=1e-3, patience=5, early_stopping=True):
    # Create unique timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_log_dir = os.path.join(log_dir, f'run_{timestamp}')
    run_save_dir = os.path.join(save_dir, f'model_{timestamp}')
    os.makedirs(run_save_dir, exist_ok=True)
    writer = SummaryWriter(run_log_dir)

    train_loader, val_loader = get_dataloaders(data_dir, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training on {len(train_loader.dataset)} training samples and {len(val_loader.dataset)} validation samples.")
    model = EmotionRecognitionModel(num_classes=7).to(device)

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
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Time: {elapsed:.2f}s")
        writer.add_scalars('Loss', {'Train': train_loss, 'Val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'Train': train_acc, 'Val': val_acc}, epoch)
        writer.add_scalar('Time/Epoch', elapsed, epoch)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(run_save_dir, 'best_model.pth'))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if early_stopping and epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    writer.close()
    print("Training complete.")


if __name__ == '__main__':
    train_model(data_dir='data')
