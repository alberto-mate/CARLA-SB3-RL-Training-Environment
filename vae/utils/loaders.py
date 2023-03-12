""" Some data loading utilities """
import os
import random

from PIL import Image
from torch.utils.data import Dataset, TensorDataset, random_split
from torchvision import transforms


class ImagePairDataset(Dataset):
    def __init__(self, folder_a, folder_b):
        self.folder_a = folder_a
        self.folder_b = folder_b
        self.image_names = os.listdir(folder_a)
        self.transform_a = transforms.Compose(
            [transforms.ToTensor()])
        self.transform_b = transforms.Compose(
            [transforms.ToTensor()])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_a = Image.open(os.path.join(self.folder_a, image_name))
        image_b = Image.open(os.path.join(self.folder_b, image_name))

        image_a = self.transform_a(image_a)
        image_b = self.transform_b(image_b)

        return image_a, image_b


class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        # Return lambda function if transform is None else return transform
        self.transform = transform if transform else lambda x: x

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)
