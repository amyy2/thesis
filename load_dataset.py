import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import nibabel as nib

class BraTSDataset(Dataset):
    def __init__(self, path):
        self.path = path
    
    def __len__(self):
        return len(os.listdir(self.path))
    
    def __getitem__(self, index):
        folder = os.listdir(self.path)[index]
        folder_path = self.path + '/' + folder + '/' + folder
        image = np.stack(
            nib.load(folder_path + '_flair.nii').get_fdata(),
            nib.load(folder_path + '_t1.nii').get_fdata(),
            nib.load(folder_path + '_t1ce.nii').get_fdata(),
            nib.load(folder_path + '_t2.nii').get_fdata()
        )
        mask = nib.load(folder_path + '_seg.nii').get_fdata()

        return image, mask

if __name__ == "__main__":
    dataset = BraTSDataset('archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData')
    plt.imshow(dataset[0][0][0][60])
    plt.show()