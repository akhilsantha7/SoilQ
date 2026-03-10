"""
Plant disease detection – high-accuracy training with ResNet50.
Uses train/val split, two-phase fine-tuning, and saves best model by validation accuracy.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from torchvision import datasets, transforms, models
from collections import defaultdict
from torchvision.models import ResNet50_Weights
import torch.nn as nn
from datetime import datetime
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# ImageNet normalization (required for pretrained ResNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training config (tuned for higher accuracy with small data)
BATCH_SIZE = 24
VAL_RATIO = 0.15
EPOCHS_PHASE1 = 20   # classifier only
EPOCHS_PHASE2 = 30   # full fine-tune
LR_PHASE1 = 1e-3
LR_PHASE2 = 5e-5
LABEL_SMOOTHING = 0.1

# Stronger augmentation to get more out of limited data
_train_aug = [
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.15),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
try:
    _train_aug.insert(-2, transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)))
except Exception:
    pass
train_transform = transforms.Compose(_train_aug)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class SubsetWithTransform(Dataset):
    """Subset that applies a specific transform (so train and val can use different transforms)."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label


# Use augmented dataset if you ran generate_augmented_dataset.py (more images per class)
DATA_DIR = "dataset_augmented" if os.path.isdir("dataset_augmented") else "dataset"
# Load full dataset (no transform)
full_dataset = datasets.ImageFolder(DATA_DIR, transform=None)
targets = full_dataset.targets
n_total = len(full_dataset)


def stratified_split(indices_by_class, val_ratio, seed=42):
    """Train/val indices so each split keeps roughly the same class distribution."""
    g = torch.Generator().manual_seed(seed)
    train_idx, val_idx = [], []
    for cls, idx in indices_by_class.items():
        n = len(idx)
        n_val = max(1, min(n - 1, int(n * val_ratio)))
        perm = torch.randperm(n, generator=g)
        for i in range(n_val):
            val_idx.append(idx[perm[i].item()])
        for i in range(n_val, n):
            train_idx.append(idx[perm[i].item()])
    return train_idx, val_idx


idx_by_class = defaultdict(list)
for i, t in enumerate(targets):
    idx_by_class[t].append(i)
train_indices, val_indices = stratified_split(idx_by_class, VAL_RATIO)
n_train, n_val = len(train_indices), len(val_indices)

train_subset = Subset(full_dataset, train_indices)
val_subset = Subset(full_dataset, val_indices)
train_dataset = SubsetWithTransform(train_subset, train_transform)
val_dataset = SubsetWithTransform(val_subset, val_transform)

num_classes = len(full_dataset.class_to_idx)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = device.type == "cuda"

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin_memory)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Device: {device} | Classes: {num_classes} | Train: {n_train} | Val: {n_val}")


def build_resnet50(num_classes):
    """ResNet50 with pretrained ImageNet weights and custom classifier."""
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes),
    )
    return model


def accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            out = model(inputs)
            _, pred = out.max(1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)
    return correct / total if total else 0.0


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        out = model(inputs)
        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


def run_phase(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, phase_name):
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    for e in range(epochs):
        t0 = datetime.now()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = accuracy(model, val_loader)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                out = model(inputs)
                val_loss += criterion(out, targets).item()
        val_loss /= len(val_loader)
        if scheduler is not None:
            scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        dt = (datetime.now() - t0).total_seconds()
        print(f"{phase_name} Epoch {e+1}/{epochs} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | {dt:.0f}s")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "plant_disease_model_1.pt")
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")
    return history


# Build model
model = build_resnet50(num_classes).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

# Phase 1: train classifier only
optimizer1 = torch.optim.Adam(model.fc.parameters(), lr=LR_PHASE1)
scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode="min", factor=0.5, patience=2)
print("\n--- Phase 1: Training classifier only ---")
h1 = run_phase(model, train_loader, val_loader, criterion, optimizer1, scheduler1, EPOCHS_PHASE1, "Phase1")

# Phase 2: unfreeze backbone, fine-tune with small LR
for p in model.parameters():
    p.requires_grad = True
optimizer2 = torch.optim.AdamW(model.parameters(), lr=LR_PHASE2, weight_decay=1e-4)
scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode="min", factor=0.5, patience=3)
print("\n--- Phase 2: Full fine-tuning ---")
h2 = run_phase(model, train_loader, val_loader, criterion, optimizer2, scheduler2, EPOCHS_PHASE2, "Phase2")

# Load best checkpoint and report
model.load_state_dict(torch.load("plant_disease_model_1.pt"))
model.eval()
final_train_acc = accuracy(model, train_loader)
final_val_acc = accuracy(model, val_loader)
print(f"\nBest model – Train accuracy: {final_train_acc:.4f} | Val accuracy: {final_val_acc:.4f}")

# Plots
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].plot(h1["train_loss"] + h2["train_loss"], label="train_loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].set_title("Train loss")
axes[1].plot(h1["val_loss"] + h2["val_loss"], label="val_loss", color="orange")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].set_title("Val loss")
axes[2].plot(h1["val_acc"] + h2["val_acc"], label="val_acc", color="green")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Accuracy")
axes[2].legend()
axes[2].set_title("Val accuracy")
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()
