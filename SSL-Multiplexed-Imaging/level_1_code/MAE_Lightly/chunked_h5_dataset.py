from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io

import torch
from torchvision import transforms
from torch.utils.data import Dataset

import h5py

def decode_name(name):
    # If it's an integer id, just return str(id)
    if isinstance(name, (int, np.integer)):
        return str(int(name))

    # If it's bytes
    if isinstance(name, (bytes, np.bytes_)):
        return name.decode("utf-8")

    # If it's an array/scalar container (common in h5py)
    try:
        arr = np.array(name).reshape(-1)
        if arr.size == 0:
            return ""
        v = arr[0]
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (bytes, np.bytes_)):
            return v.decode("utf-8")
        return str(v)
    except Exception:
        return str(name)

class hdf5_multichannel_singleh5():
    def __init__(self, file_path):
        self.h5_file = h5py.File(file_path, 'r')
        self.images = self.h5_file['Images']
        self.image_names = self.h5_file['Names']

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,i):
        image = self.images[i] # [8, 256, 256]
        # image = np.transpose(image, (2, 0, 1)) # [256, 256, 8]
        name = self.image_names[i]
        return image, decode_name(name)

class h5_chunk_wrapper(Dataset):
    def __init__(self, h5_dir, transform=None):
        h5_dir = Path(h5_dir)
        # get files
        files = sorted(list(h5_dir.glob('*.h5')))
        # open h5 files
        self.single_h5s = [hdf5_multichannel_singleh5(pfile) for pfile in files]
        # register
        self.register = {}
        idx = 0
        for h5id, h5class in enumerate(self.single_h5s):
            for imgid in range(h5class.__len__()):
                self.register[idx] = [h5id, imgid]
                idx += 1
              
        # define transform
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([])
            
    def __len__(self):
        return len(self.register)
    
    def __getitem__(self, i):
        h5id, imgid = self.register[i]
        image, name = self.single_h5s[h5id].__getitem__(imgid)
        image = torch.from_numpy(image)
        image = self.transform(image)
        return image, decode_name(name)
    

BOMI_meanstds = {  
                                'DAPI' : [0.864308, 0.577803],
                                'CD4': [0.709685, 0.767798],
                                'CD20': [0.105369, 0.379986],
                                'CD8': [0.172058, 0.348507],
                                'FoxP3': [0.140908, 0.174657],
                                'CD45RO': [0.368375, 0.360702],
                                'PanCK': [1.403302, 0.985561],
                                'Autofluorescence': [1.473905, 0.489961]

                                }


BOMI_CHANNEL_NAMES = [
    "DAPI",
    "CD4",
    "CD20",
    "CD8",
    "FoxP3",
    "CD45RO",
    "PanCK",
    "Autofluorescence",
]




def get_mean_std(args=None):
    channel_names = [
        "DAPI", "CD4", "CD20", "CD8",
        "FoxP3", "CD45RO", "PanCK", "Autofluorescence"
    ]

    ds_mean = np.array([BOMI_meanstds[k][0] for k in channel_names], dtype=np.float32)
    ds_std  = np.array([BOMI_meanstds[k][1] for k in channel_names], dtype=np.float32)

    return {"mean": ds_mean, "std": ds_std}, len(channel_names)

mean_std, nC = get_mean_std()
print("n_channels:", nC, "mean shape:", mean_std["mean"].shape)
