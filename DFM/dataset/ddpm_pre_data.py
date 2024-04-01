import h5py
import os
from scipy.ndimage.interpolation import zoom
import torch
from torch.utils.data import Dataset

class ddpm_pre_ACDCDataset(Dataset):
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

        x, y = img.shape
        img = zoom(img, (self.image_size / x, self.image_size / y), order=0)   
        img = torch.from_numpy(img).unsqueeze(0).float()  
        img_name = str(id.split('/')[2].split('.')[0])

        return img, img_name  

    def __len__(self):
        return len(self.ids)
