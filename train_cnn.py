import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights
from torch.utils.data import DataLoader

device = torch.device("cpu")

# ========================
# DATA
# ========================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

train_dataset = datasets.ImageFolder(
    "dataset2/train",
    transform=transform
)

val_dataset = datasets.ImageFolder(
    "dataset2/val",
    transform=transform
)

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
# MODEL EfficientNet
# ========================
model = models.efficientnet_b0(
    weights=EfficientNet_B0_Weights.DEFAULT
)

# Đổi classifier thành 2 class
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)

model.to(device)

# ========================
# LOSS + OPTIMIZER
# ========================
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)

# ========================
# TRAIN SETUP
# ========================
epochs = 3

train_losses = []
val_losses = []

train_accs = []
val_accs = []

best_val_acc = 0

# ========================
# TRAIN LOOP
# ========================
for epoch in range(epochs):

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

        _, predicted = torch.max(outputs,1)

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

            _, predicted = torch.max(outputs,1)

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
            "efficientnet_best.pth"
        )

        print("Best model saved!")

# ========================
# FINAL SAVE
# ========================
torch.save(
    model.state_dict(),
    "efficientnet_last.pth"
)

print("EfficientNet model saved!")

# ========================
# PLOT LOSS
# ========================
plt.figure()

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")

plt.legend()

plt.title("EfficientNet Loss Curve")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.savefig("efficientnet_loss_curve.png")

plt.show()

# ========================
# PLOT ACCURACY
# ========================
plt.figure()

plt.plot(train_accs, label="Train Accuracy")
plt.plot(val_accs, label="Val Accuracy")

plt.legend()

plt.title("EfficientNet Accuracy Curve")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.savefig("efficientnet_accuracy_curve.png")

plt.show()

# ========================
# PRINT BEST RESULT
# ========================
print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")