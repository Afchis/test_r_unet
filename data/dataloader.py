import os
import numpy as np

import torch
from torch.utils.data import Dataset, Dataloader
import torchvision
import torchvision.transforms as transforms

from PIL import Image



class MedData(Dataset):
    def __init__(self, timesteps=TIMESTEPS, folder_data=FOLDER_DATA, folder_mask=FOLDER_MASK, 
    			 file_names = FILE_NAMES):
        super().__init__()
        self.time = timesteps
        self.folder_data = folder_data
        self.folder_mask = folder_mask
        self.file_names = file_names

        self.transform = transforms.Compose([
                                             transforms.Resize((128, 128), interpolation = 0),
                                             transforms.ToTensor()
                                             ])
        self.to_tensor = transforms.ToTensor()


    def __getitem__(self, idx):
        gif_list = []
        for i in range(self.time):
            gif_list.append(self.transform(Image.open(self.folder_data + '/' + file_names[idx+i])).unsqueeze(0))
        gif_data = torch.stack(gif_list)
        gif_list.clear()
        for i in range(self.time):
            gif_list.append(self.transform(Image.open(self.folder_mask + '/' + file_names[idx+i])).unsqueeze(0))
        gif_mask = torch.stack(gif_list)
        gif_list.clear()
        for i in range(self.time):
            img = Image.open(folder_mask + '/' + file_names[idx+i])
            img = img.resize((128, 128), resample=Image.NEAREST)
            gif_list.append(self.to_tensor(morph.distance_transform_edt(np.asarray(img)/255)).unsqueeze(0))
        gif_depth = torch.stack(gif_list)
        return gif_data, gif_mask, gif_depth
    
    def __len__(self):
        return len(self.file_names) - self.time + 1