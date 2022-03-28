import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import datasets

from config import data_transforms

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
def ImageDataset(args=None):
    # datadir = args.dataset
    datadir = '/mnt/disk1/vaipe-data/prescription/data_matching/simulate-data-thanhnt/all_imgs_simulate_thanhnt'
    devided_data = {}
    # devided_data = {x: datasets.ImageFolder(os.path.join(datadir,x), data_transforms)
    #                   for x in ['train', 'test']}
    devided_data['train'] = datasets.ImageFolder(os.path.join(datadir,'train'), data_transforms)
    # pills_class_to_idx = devided_data['train'].class_to_idx
    devided_data['test'] = CustomImageDataset('./testset.csv', os.path.join(datadir, 'test'),
                                              transform=data_transforms)
    # devided_data['test'] = datasets.ImageFolder(os.path.join(datadir,'train'), data_transforms, target_transform=pills_class_to_idx)            
    # print(name,label)
    return devided_data
    
if __name__ == '__main__':
    ImageDataset()