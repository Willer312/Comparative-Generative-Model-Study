import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import datasets, transforms


FASHION_MNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def get_fashion_mnist_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    seed: int = 42,
    normalize_to_neg_one: bool = False,
    num_workers: int = 2,
):
    """
    Loads Fashion-MNIST (70,000 images) and creates a deterministic
    80% Train (56,000) / 10% Validation (7,000) / 10% Test (7,000) split.
    
    Args:
        normalize_to_neg_one: If False, pixels are in [0, 1] (for MLP, AE, VAE).
                              If True, pixels are in [-1, 1] (for GAN, DCGAN, DDPM).
    """
    if normalize_to_neg_one:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) 
        ])
    else:
        transform = transforms.ToTensor()  

  
    train_part = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_part = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    full_dataset = ConcatDataset([train_part, test_part])

   
    total_size = len(full_dataset) 
    train_size = int(0.8 * total_size)  
    val_size = int(0.1 * total_size)    

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_size, generator=generator).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader