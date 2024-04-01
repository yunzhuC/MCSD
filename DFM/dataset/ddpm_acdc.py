import h5py
import numpy as np
import os
from scipy.ndimage.interpolation import zoom
import torch
from torch.utils.data import Dataset

class ddpm_ACDCDataset(Dataset):
    def __init__(self, data_dir, image_size, num_images, id_path=None):
        self.data_dir = data_dir
        self.id_path = id_path
        self.image_size = image_size
        self.num_images = num_images

        if num_images > 0:
            print(f"Take first {num_images} images...")

        with open(id_path, 'r') as f:
            self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        sample = h5py.File(os.path.join(self.data_dir, id), 'r')
        img = sample['image'][:]
        mask = sample['label'][:]
        x, y = img.shape
        img = zoom(img, (self.image_size / x, self.image_size / y), order=0)   
        mask = zoom(mask, (self.image_size / x, self.image_size / y), order=0)

        img = torch.from_numpy(img).unsqueeze(0).float()  
        mask = torch.from_numpy(np.array(mask)).long()  

        return img, mask

    def __len__(self):
        return len(self.ids)
