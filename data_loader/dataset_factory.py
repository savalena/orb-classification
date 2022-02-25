from torch.utils.data import DataLoader
from .orb_features_dataset import ORBFeaturesDataset, DEFAULT_TRANSFORM


class DatasetFactory:
    def __init__(self,
                 root,
                 dataset,
                 labels_file,
                 transform=DEFAULT_TRANSFORM,
                 batch_size=128,
                 shuffle=True,
                 num_workers=4):
        self.root = root
        self.dataset = dataset
        self.labels_file = labels_file
        self.transform = transform

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def make_data_loader(self):
        return DataLoader(ORBFeaturesDataset(self.root,
                                             self.dataset,
                                             self.labels_file,
                                             transform=self.transform),
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers)
