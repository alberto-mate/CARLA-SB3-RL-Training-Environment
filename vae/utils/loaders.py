""" Some data loading utilities """
import os
from PIL import Image
from torch.utils.data import Dataset


class ImagePairDataset(Dataset):
    def __init__(self, folder_a, folder_b, transform):
        self.folder_a = folder_a
        self.folder_b = folder_b
        self.image_names = os.listdir(folder_a)
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_a = Image.open(os.path.join(self.folder_a, image_name))
        image_b = Image.open(os.path.join(self.folder_b, image_name))

        image_a = self.transform(image_a)
        image_b = self.transform(image_b)

        return image_a, image_b
