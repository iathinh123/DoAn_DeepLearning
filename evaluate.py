import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)

import timm

# ========================
# DEVICE
# ========================
device = torch.device("cpu")

# ========================
# DATA
# ========================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

test_dataset = datasets.ImageFolder(
    r"D:\DoAn_DeepLearning\dataset2\test",
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32
)

# ========================
# CONFUSION MATRIX
# ========================
def plot_confusion_matrix(cm, title):

    plt.figure(figsize=(6,6))

    plt.imshow(cm)

    plt.title(title)

    plt.colorbar()

    classes = [
        "Non-Recyclable",
        "Recyclable"
    ]

    plt.xticks([0,1], classes)
    plt.yticks([0,1], classes)

    for i in range(2):
        for j in range(2):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()

# ========================
# HYBRID MODEL
# ========================
class HybridModel(nn.Module):

    def __init__(self):

        super(HybridModel, self).__init__()

        # ========================
        # ViT
        # ========================
        self.vit = timm.create_model(
            'vit_small_patch16_224',
            pretrained=False,
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

        vit_output = self.vit(x)

        resnet_output = self.resnet(x)

        final_output = (
            0.7 * vit_output +
            0.3 * resnet_output
        )

        return final_output

# ========================
# EVALUATE FUNCTION
# ========================
results = []

def evaluate_model(model, name):

    model.to(device)

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs,1)

            y_true.extend(labels.numpy())

            y_pred.extend(
                predicted.cpu().numpy()
            )

    acc = accuracy_score(
        y_true,
        y_pred
    )

    print("\n============================")
    print(f"MODEL: {name}")
    print("============================")

    print(f"Accuracy: {acc:.4f}")

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "Non-Recyclable",
                "Recyclable"
            ]
        )
    )

    # ========================
    # CONFUSION MATRIX
    # ========================
    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plot_confusion_matrix(
        cm,
        f"{name} - Confusion Matrix"
    )

    # ========================
    # SAVE RESULT
    # ========================
    results.append(
        (name, acc)
    )

# ========================
# LOAD HYBRID MODEL
# ========================
hybrid_model = HybridModel()

hybrid_model.load_state_dict(
    torch.load(
        r"D:\DoAn_DeepLearning\models\hybrid_vit_resnet18_best.pth",
        map_location=device
    )
)

# ========================
# LOAD EFFICIENTNET
# ========================
efficient_model = models.efficientnet_b0()

efficient_model.classifier[1] = nn.Linear(
    efficient_model.classifier[1].in_features,
    2
)

efficient_model.load_state_dict(
    torch.load(
        r"D:\DoAn_DeepLearning\models\efficientnet_best.pth",
        map_location=device
    )
)

# ========================
# EVALUATE
# ========================
evaluate_model(
    hybrid_model,
    "Hybrid ViT + ResNet18"
)

evaluate_model(
    efficient_model,
    "EfficientNet"
)

# ========================
# SUMMARY
# ========================
print("\n===== SUMMARY =====")

for name, acc in results:

    print(
        f"{name}: {acc:.4f}"
    )