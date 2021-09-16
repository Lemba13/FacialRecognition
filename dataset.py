import pandas as pd
import cv2
from PIL import Image,ImageOps
from torch.utils.data import Dataset, DataLoader
import config
import torch
import numpy as np



class FRDataset(Dataset):
    def __init__(self,csv_file,transform=None):
        self.transform=transform
        self.path=pd.read_csv(csv_file)
    
    def __len__(self):
        return len(self.path)
    
    def __getitem__(self,index):
        img1_path=self.path.iloc[index,1]
        img2_path=self.path.iloc[index,2]
        img3_path = self.path.iloc[index, 3]
        
        img1 = Image.open(img1_path)
        img1 = np.asarray(ImageOps.grayscale(img1))/255.0
        
        img2 = Image.open(img2_path)
        img2 = np.asarray(ImageOps.grayscale(img2))/255.0
        
        img3 = Image.open(img3_path)
        img3 = np.asarray(ImageOps.grayscale(img3))/255.0

        
        if self.transform:
            augmentations1=self.transform(image=img1)
            img1=augmentations1['image']
            augmentations2 = self.transform(image=img2)
            img2 = augmentations2['image']
            augmentations3 = self.transform(image=img3)
            img3 = augmentations3['image']
        
        return img1.type(torch.FloatTensor), img2.type(torch.FloatTensor), img3.type(torch.FloatTensor) 
    
    
class TestDataset(Dataset):
    def __init__(self, csv_file_path, transform=None):
        self.transform = transform
        self.path = pd.read_csv(csv_file_path)

    def __len__(self):
        return len(self.path)

    def __getitem__(self, index):
        img1_path = self.path.iloc[index, 0]
        img2_path = self.path.iloc[index, 1]
        label = self.path.iloc[index, 2]

        img1 = Image.open(img1_path)
        img1 = np.asarray(ImageOps.grayscale(img1))/255.0

        img2 = Image.open(img2_path)
        img2 = np.asarray(ImageOps.grayscale(img2))/255.0

        if self.transform:
            augmentations1 = self.transform(image=img1)
            img1 = augmentations1['image']
            augmentations2 = self.transform(image=img2)
            img2 = augmentations2['image']

        return img1.type(torch.FloatTensor), img2.type(torch.FloatTensor), label

def test():
    
    dataset=FRDataset(
        'csv/imgs_triplet.csv',
        transform=config.transform
    )
    
    sampleloader=DataLoader(dataset=dataset, batch_size=2, shuffle=True)
    
    print(next(iter(sampleloader))[0].max())
   
if __name__=='__main__':
    test()
