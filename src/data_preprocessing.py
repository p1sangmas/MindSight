import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_dataloaders(data_dir='data', batch_size=64):
    transform_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # FER2013 is grayscale
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader
