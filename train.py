import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader

from sklearn.utils.class_weight import compute_class_weight

import timm

# ========================
# DEVICE
# ========================
device = torch.device("cpu")

# ========================
# DATA PREPROCESSING
# ========================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ========================
# DATASET
# ========================
train_dataset = datasets.ImageFolder(
    r"D:\DoAn_DeepLearning\dataset2\train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    r"D:\DoAn_DeepLearning\dataset2\val",
    transform=test_transform
)

# ========================
# DATALOADER
# ========================
train_loader = DataLoader(
    train_dataset,
    batch_size=96,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=96
)

    # ========================
# HANDLE IMBALANCED DATA
# ========================
labels = [label for _, label in train_dataset.samples]

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float
).to(device)

# ========================
# HYBRID MODEL
# ViT + ResNet18
# ========================
class HybridModel(nn.Module):

    def __init__(self):
        super(HybridModel, self).__init__()

        # ========================
        # ViT
        # ========================
        self.vit = timm.create_model(
            'vit_small_patch16_224',
            pretrained=True,
            num_classes=2
        )

        # ========================
        # ResNet18
        # ========================
        self.resnet = models.resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        self.resnet.fc = nn.Linear(
            self.resnet.fc.in_features,
            2
        )

    def forward(self, x):

        # Output từ ViT
        vit_output = self.vit(x)

        # Output từ ResNet18
        resnet_output = self.resnet(x)

        # Weighted Ensemble
        final_output = (
            0.7 * vit_output +
            0.3 * resnet_output
        )

        return final_output

# ========================
# LOAD MODEL
# ========================
model = HybridModel()
model.to(device)

# ========================
# LOSS FUNCTION
# ========================
criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

# ========================
# OPTIMIZER
# ========================
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.0001
)

# ========================
# TRAINING SETUP
# ========================
epochs = 3

train_losses = []
val_losses = []

train_accs = []
val_accs = []

best_val_acc = 0

# ========================
# TRAINING LOOP
# ========================
for epoch in range(epochs):

    # ========================
    # TRAIN MODE
    # ========================
    model.train()

    running_loss = 0
    correct = 0
    total = 0

    print(f"\nEpoch {epoch+1}/{epochs}")

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        if batch_idx % 10 == 0:
            print(
                f"Batch {batch_idx}/{len(train_loader)} "
                f"- Loss: {loss.item():.4f}"
            )

    train_acc = 100 * correct / total

    train_losses.append(running_loss)
    train_accs.append(train_acc)

    print(f"Train Loss: {running_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.2f}%")

    # ========================
    # VALIDATION
    # ========================
    model.eval()

    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total

    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Accuracy: {val_acc:.2f}%")

    # ========================
    # SAVE BEST MODEL
    # ========================
    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            "hybrid_vit_resnet18_best.pth"
        )

        print("Best model saved!")

# ========================
# FINAL SAVE
# ========================
torch.save(
    model.state_dict(),
    "hybrid_vit_resnet18_last.pth"
)

print("Final model saved!")

# ========================
# PLOT LOSS CURVE
# ========================
plt.figure()

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")

plt.legend()

plt.title("Loss Curve")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.savefig("loss_curve.png")

plt.show()

# ========================
# PLOT ACCURACY CURVE
# ========================
plt.figure()

plt.plot(train_accs, label="Train Accuracy")
plt.plot(val_accs, label="Val Accuracy")

plt.legend()

plt.title("Accuracy Curve")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.savefig("accuracy_curve.png")

plt.show()

# ========================
# PRINT BEST RESULT
# ========================
print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")