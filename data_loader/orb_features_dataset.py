from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
import torch
import numpy as np
import os
from pathlib import Path

DEFAULT_RESNET_TRANSFORM = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


class ORBFeaturesDataset(Dataset):
    def __init__(self, root, datasets, labels_file, classification_threshold, transform):
        root = Path(root)
        images_list = []
        labels_list = []
        for dataset in datasets:
            print('datasetname:', dataset)
            dataset_path = root / dataset
            images_path = dataset_path / 'images'
            dataset_images = np.array([str(images_path / image) for image in np.sort(os.listdir(images_path))])
            assert len(dataset_images) > 0
            images_list.append(dataset_images)

            dataset_labels = pd.read_csv(str(dataset_path / labels_file), header=None).to_numpy().reshape(-1)
            assert dataset_labels.shape[0] > 0
            labels_list.append(dataset_labels > classification_threshold)
            print('num images:', dataset_images.shape[0])
            print('num labels:',  dataset_labels.shape[0])
            assert dataset_images.shape[0] == dataset_labels.shape[0]

        self.images_list = np.concatenate(images_list, axis=0)
        self.labels = np.concatenate(labels_list, axis=0)
        self.transform = transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = read_image(self.images_list[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label
