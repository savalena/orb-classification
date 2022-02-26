from .base_model import BaseModel
import torch
import torch.nn as nn
from torchvision import transforms

DEFAULT_RESNET_TRANSFORM = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


class Resnet18Model(BaseModel):
    def __init__(self, fc, transform=DEFAULT_RESNET_TRANSFORM, freeze_backbone=False):
        super().__init__(transform)
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.backbone.fc = nn.Identity()
        self.fc = fc
        self.prepare_backbone(freeze_backbone)

    def forward(self, x):
        y = self.backbone(x)
        return self.fc(y)
