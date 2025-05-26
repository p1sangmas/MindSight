import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from collections import Counter
from torch.utils.data import WeightedRandomSampler

def get_dataloaders(data_dir='data', batch_size=64, oversample=False):
    transform_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # FER2013 is grayscale
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    transform_val = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'test')

    train_dataset = datasets.ImageFolder(train_path, transform=transform_train)
    val_dataset = datasets.ImageFolder(val_path, transform=transform_val)

    if oversample:
        # Compute class counts for oversampling
        targets = [sample[1] for sample in train_dataset.samples]
        class_counts = Counter(targets)
        num_classes = len(class_counts)
        class_sample_count = [class_counts.get(i, 0) for i in range(num_classes)]
        weights = 1. / torch.tensor(class_sample_count, dtype=torch.float)
        sample_weights = [weights[label] for label in targets]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader
