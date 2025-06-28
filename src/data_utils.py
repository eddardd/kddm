import os
import torch
from typing import Tuple
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset


def get_cifar10_dataloaders(batch_size=128):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408),
                (0.2675, 0.2565, 0.2761)
            ),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408),
                (0.2675, 0.2565, 0.2761)
            ),
        ]
    )

    # Loading the CIFAR-10 dataset:
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader


def get_svhn_dataloaders(
    batch_size: int = 128
) -> Tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4377, 0.4438, 0.4728),
            (0.1980, 0.2010, 0.1970)
        )
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4377, 0.4438, 0.4728),
            (0.1980, 0.2010, 0.1970)
        )
    ])

    # Loading SVHN dataset
    train_dataset = datasets.SVHN(
        root='./data',
        split='train',
        download=True,
        transform=train_transform
    )
    test_dataset = datasets.SVHN(
        root='./data',
        split='test',
        download=True,
        transform=test_transform
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader


def get_caltech256_dataloaders(
    batch_size: int = 128,
    image_size: int = 224,
    split_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, list]:
    """
    Load Caltech-256 dataset from existing directory structure with automatic
    splitting.

    Directory structure expected:
    ./data/caltech256/256_ObjectCategories/
        ├── class001/
        ├── class002/
        └── .../

    Args:
        batch_size: Batch size for dataloaders
        image_size: Target image size (will be center cropped)
        split_ratio: Ratio of training data (0.8 = 80% train, 20% test)
        seed: Random seed for reproducible splits
    """
    # ImageNet-style normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.3, 0.3, 0.3),  # type: ignore
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.2)  # Regularization
    ])

    test_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.2)),  # Slightly larger resize
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load dataset using ImageFolder (since we have directory structure)
    dataset_path = os.path.join('./data/caltech256', '256_ObjectCategories')
    full_dataset = datasets.ImageFolder(
        root=dataset_path,
        transform=None  # We'll apply transforms after split
    )

    # Create stratified split (maintains class distribution)
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(
        list(range(len(full_dataset))),
        test_size=1-split_ratio,
        random_state=seed,
        stratify=full_dataset.targets
    )

    # Create subset wrappers with transforms
    class TransformSubset(Subset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, y

        def __len__(self):
            return len(self.subset)

    train_dataset = TransformSubset(
        Subset(full_dataset, train_idx),
        train_transform
    )
    test_dataset = TransformSubset(
        Subset(full_dataset, test_idx),
        test_transform
    )

    # Create dataloaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )

    # Print dataset statistics
    print("\nCaltech-256 Dataset:")
    print(f"Total images: {len(full_dataset)}")
    print(f"Classes: {len(full_dataset.classes)}")
    print(f"Training images: {len(train_dataset)} ({split_ratio*100:.1f}%)")
    print(f"Test images: {len(test_dataset)} ({100-split_ratio*100:.1f}%)")
    print(f"Sample shape: {next(iter(train_loader))[0].shape}\n")

    return train_loader, test_loader, full_dataset.classes


def get_cifar100_dataloaders(
    batch_size: int = 128
) -> Tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408),
                (0.2675, 0.2565, 0.2761)
            ),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408),
                (0.2675, 0.2565, 0.2761)
            ),
        ]
    )

    # Loading the CIFAR-10 dataset:
    train_dataset = datasets.CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )
    test_dataset = datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader


def get_mnist_dataloaders(
    batch_size: int = 128
) -> Tuple[DataLoader, DataLoader]:
    # Train transformations
    train_transform = transforms.Compose([
        # Small rotations and translations
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        # Slight zoom-in and zoom-out
        transforms.RandomResizedCrop(28, scale=(0.9, 1.1)),
        # Convert to tensor (shape: [1, 28, 28])
        transforms.ToTensor(),
        # Normalize with dataset mean and std
        transforms.Normalize((0.1307,), (0.3081,)),
        # Flatten to 1D vector (shape: [784])
        transforms.Lambda(lambda x: x.view(-1))
    ])

    # Test transformations
    test_transform = transforms.Compose([
        # Convert to tensor (shape: [1, 28, 28])
        transforms.ToTensor(),
        # Only normalize, no data augmentation
        transforms.Normalize((0.1307,), (0.3081,)),
        # Flatten to 1D vector (shape: [784])
        transforms.Lambda(lambda x: x.view(-1))
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=test_transform
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader
