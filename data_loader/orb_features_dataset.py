from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
import torch
import numpy as np
import os
from pathlib import Path

DEFAULT_TRANSFORM = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            # transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


class ORBFeaturesDataset(Dataset):
    def __init__(self, root, dataset, labels_file, classification_threshold, transform=DEFAULT_TRANSFORM):
        root = Path(root)
        self.dataset_path = root / dataset
        images_path = self.dataset_path / 'images'
        self.images_list = np.array([str(images_path / image) for image in os.listdir(images_path)])
        assert len(self.images_list) > 0
        self.labels = pd.read_csv(str(self.dataset_path / labels_file))

        self.transform = transform

        self.classification_threshold = classification_threshold

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = read_image(self.images_list[idx])
        label = torch.tensor(self.labels.iloc[idx][0], dtype=torch.float)
        label = label > self.classification_threshold

        if self.transform:
            image = self.transform(image)

        return image, label
