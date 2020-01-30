from data.dataloader import MedData
from models.rec_small_unet import UNetSmall

import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


TIMESTEPS = 3
BATCH_SIZE = 4
NUM_EPOCHS = 25
INPUT_SIZE = 128

GRU_NAN = False

FILE_NAMES = sorted(os.listdir("../test_r_unet/data/images"))

FOLDER_DATA = "../test_r_unet/data/images"
FOLDER_MASK = "../test_r_unet/data/labels"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = MedData(timesteps=TIMESTEPS, folder_data=FOLDER_DATA,
				  folder_mask=FOLDER_MASK, file_names=FILE_NAMES)

train_loader = DataLoader(dataset=dataset,
						  batch_size=BATCH_SIZE,
						  num_workers=2,
						  shuffle=False
						  )

model = UNetSmall(batch_size=BATCH_SIZE, timesteps=TIMESTEPS, gru_nan=GRU_NAN)
print(model)