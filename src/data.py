""" Script to download and preprocess the dataset """

import os
import zipfile
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, random_split

class Dataset_Setup:
    def __init__(self, data_dir, url, root_dir = './'):
        self.data_dir = data_dir
        self.url = url
        self.root_dir = root_dir
        self.download()
        self.extract()

    def download(self):
        if os.path.exists(os.path.join(self.root_dir, 'dataset.zip')):
            print("Dataset already downloaded")
            return
        
        print("Downloading dataset...")
        os.system(f"wget {self.url} -O {self.root_dir}/dataset.zip -P {self.data_dir}")
        print("Download complete.")
    
    def extract(self):
        print("Extracting dataset...")
        os.makedirs(self.data_dir, exist_ok=True)
        with zipfile.ZipFile(os.path.join(self.root_dir, "dataset.zip"), "r") as zip_ref:
            zip_ref.extractall(self.data_dir)
        print("Extraction complete.")



class AlzDataset(Dataset):
    def __init__(
            self, 
            data_dir, 
            split = {'train': True, 'val': True, 'test': True}, 
            split_ratio = {'train': 0.8, 'val': 0.1, 'test': 0.1}, 
            transform=None
            ):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        self.split_ratio = split_ratio

        self.paths = []
        for class_folder in os.listdir(self.data_dir):
            class_dir = os.path.join(self.data_dir, class_folder)
            if os.path.isdir(class_dir):
                for img in os.listdir(class_dir):
                    sample_path = os.path.join(class_dir, img)
                    self.paths.append(sample_path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_name = self.paths[idx]
        image = Image.open(img_name)
        image = image.resize((64, 64), Image.ANTIALIAS)
        image = np.asarray(image)
        if self.transform:
            image = self.transform(image)
        label = 1 if "AD_Data" in self.paths else 0

        return image, label
    
    def split_dataset(self):
        train_size = int(len(self)*self.split_ratio['train'])
        val_size = int(len(self)*self.split_ratio['val'])
        split_dataset = random_split(self, [train_size, 
                                            val_size, 
                                            len(self)-train_size-val_size])
        return split_dataset[0], split_dataset[1], split_dataset[2]