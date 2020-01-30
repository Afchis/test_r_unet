import os


if __name__ == "__main__":
TIMESTEPS = 3
BATCH_SIZE = 4
NUM_EPOCHS = 25
INPUT_SIZE = 128

GRU_NAN = False

FILE_NAMES = sorted(os.listdir("../test_r_unet/data/images"))

FOLDER_DATA = "../test_r_unet/data/images"
FOLDER_MASK = '../test_r_unet/data/labels'