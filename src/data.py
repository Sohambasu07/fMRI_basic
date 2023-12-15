""" Script to download and preprocess the dataset """

import os
import zipfile
from PIL import Image
from torch.utils.data import Dataset, random_split

class Dataset_Setup:
    def __init__(self, data_dir, url):
        self.data_dir = data_dir
        self.url = url
        self.download()
        self.extract()

    def download(self):
        if os.path.exists(self.data_dir):
            print("Dataset already downloaded")
            return
        
        os.makedirs(self.data_dir, exist_ok=True)
        print("Downloading dataset...")
        os.system(f"wget {self.url} -P {self.data_dir}")
        print("Download complete.")
    
    def extract(self):
        print("Extracting dataset...")
        with zipfile.ZipFile(os.path.join(self.data_dir, "dataset.zip"), "r") as zip_ref:
            zip_ref.extractall(self.data_dir)
        print("Extraction complete.")



class AlzDataset(Dataset):
    def __init__(
            self, 
            root_dir, 
            split = {'train': True, 'val': True, 'test': True}, 
            split_ratio = {'train': 0.8, 'val': 0.1, 'test': 0.1}, 
            transform=None
            ):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = os.listdir(self.root_dir)
        self.split = split
        self.split_ratio = split_ratio

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.file_list[idx])
        image = Image.open(img_name)
        image = image.resize((64, 64), Image.ANTIALIAS)

        if self.transform:
            image = self.transform(image)
        label = 1 if "AD_Data" in self.root_dir else 0

        return image, label
    
    def split_dataset(self):
        train_size = int(len(self)*self.split_ratio['train'])
        val_size = int(len(self)*self.split_ratio['val'])
        split_dataset = random_split(self, [train_size, 
                                            val_size, 
                                            len(self)-train_size-val_size])
        return split_dataset[0], split_dataset[1], split_dataset[2]