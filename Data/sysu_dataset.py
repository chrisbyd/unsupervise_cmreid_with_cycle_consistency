import os.path
from PIL import Image
import numpy as np
import torch.utils.data as data

class SYSUDataset():
    """
    This is the dataset which incorporates all the sysu train images
    """
    def __init__(self,data_dir,transform =None):

        #Load training imagess without labels
        self.train_visible_images = np.load(data_dir +'train_rgb_resized_img.npy')

        self.train_thermal_images = np.load(data_dir +'train_ir_resized_img.npy')
        self.transform= transform

    def __getitem__(self, index):
        imgA = self.train_visible_images[index]
        imgB = self.train_thermal_images[index]

        A = self.transform(imgA)
        B= self.transform(imgB)

        return A,B

    def __len__(self):
        return len(self.train_visible_images)
