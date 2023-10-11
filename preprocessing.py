import numpy as np
from sklearn.model_selection import train_test_split
from load_dataset import BraTSDataset

class DatasetCreator:
    def __init__(self):
        self.sample_names = []
    
    def load_data(self, index):
        image, mask = BraTSDataset('archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData')[index]
        return image, mask
    
    # remove samples with not enough data

    def __call__(self):
        train_samples, test_samples = train_test_split(self.sample_names, test_size=0.3, random_state=42)

if __name__ == "__main__":
    dc = DatasetCreator()
    dc.load_data(0)