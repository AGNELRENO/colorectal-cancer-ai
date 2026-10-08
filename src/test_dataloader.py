import sys
from pathlib import Path

# Allow Python to find files inside src
sys.path.append(str(Path(__file__).parent))

from dataset import ColonDataset, train_transform, val_test_transform
from split_dataset import train_samples, val_samples, test_samples

from torch.utils.data import DataLoader


# --------------------------------------------------
# Create datasets
# --------------------------------------------------

train_dataset = ColonDataset(
    train_samples,
    transform=train_transform
)

val_dataset = ColonDataset(
    val_samples,
    transform=val_test_transform
)

test_dataset = ColonDataset(
    test_samples,
    transform=val_test_transform
)


# --------------------------------------------------
# Create DataLoaders
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# Test one batch
# --------------------------------------------------

images, labels = next(iter(train_loader))


print("===== DATALOADER TEST =====")

print(f"Image batch shape : {images.shape}")
print(f"Label batch shape : {labels.shape}")

print(f"Image data type   : {images.dtype}")
print(f"Label data type   : {labels.dtype}")

print(f"First label       : {labels[0].item()}")

print("DataLoader test successful!")