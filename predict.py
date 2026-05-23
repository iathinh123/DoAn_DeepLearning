import torch
import torch.nn as nn
import timm

from torchvision import transforms, models
from PIL import Image

# ========================
# DEVICE
# ========================
device = torch.device("cpu")

# ========================
# TRANSFORM
# ========================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

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
            weights=None
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
# LOAD HYBRID MODEL
# ========================
hybrid_model = HybridModel()

hybrid_model.load_state_dict(
    torch.load(
        "models/hybrid_vit_resnet18_best.pth",
        map_location=device
    )
)

hybrid_model.to(device)

hybrid_model.eval()

# ========================
# LOAD EFFICIENTNET
# ========================
efficient_model = models.efficientnet_b0(
    weights=None
)

efficient_model.classifier[1] = nn.Linear(
    efficient_model.classifier[1].in_features,
    2
)

efficient_model.load_state_dict(
    torch.load(
        "models/efficientnet_best.pth",
        map_location=device
    )
)

efficient_model.to(device)

efficient_model.eval()

# ========================
# CLASS NAMES
# ========================
classes = [
    "Non-Recyclable",
    "Recyclable"
]

# ========================
# PREDICT FUNCTION
# ========================
def predict_image(image_path, model_name):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    # ========================
    # SELECT MODEL
    # ========================
    if model_name == "Hybrid ViT + ResNet18":

        model = hybrid_model

    else:

        model = efficient_model

    # ========================
    # PREDICT
    # ========================
    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    prediction = classes[
        predicted.item()
    ]

    confidence = confidence.item() * 100

    return prediction, confidence