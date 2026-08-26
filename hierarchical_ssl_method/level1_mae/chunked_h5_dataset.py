from pathlib import Path

import numpy as np

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

'''class hdf5_multichannel_singleh5():
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
        return image, decode_name(name)'''

class h5_chunk_wrapper(Dataset):
    def __init__(self, h5_dir, transform=None):
        self.h5_dir = Path(h5_dir)
        # get files
        self.files = sorted(self.h5_dir.glob('*.h5'))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .h5 files found in {self.h5_dir}")
        # open h5 files
        # self.single_h5s = [hdf5_multichannel_singleh5(pfile) for pfile in files]
        # register
        #self.register = {}
        #idx = 0

        file_ids = []
        local_ids = []

        for fid, fp in enumerate(self.files):
            with h5py.File(fp, 'r') as h5:
                n = len(h5['Images'])
            file_ids.append(np.full(n, fid, dtype=np.int32))
            local_ids.append(np.arange(n, dtype=np.int32))

        self.file_ids = np.concatenate(file_ids)
        self.local_ids = np.concatenate(local_ids)

        self.transform = transform if transform is not None else transforms.Compose([])

        # per-worker cache of open files
        self._cache = {}
            
    def __len__(self):
        return int(self.file_ids.shape[0])
    
    def _get_file(self, fid: int):
        if fid not in self._cache:
            h5 = h5py.File(self.files[fid], 'r')
            self._cache[fid] = h5
        return self._cache[fid]
    
    def _close_all(self):
        for h5 in self._cache.values():
            try:
                h5.close()
            except Exception:
                pass
        self._cache = {}
        
    def __del__(self):
        self._close_all()
    
    def __getitem__(self, idx):
        fid = int(self.file_ids[idx])
        lid = int(self.local_ids[idx])

        h5 = self._get_file(fid)
        img = h5["Images"][lid]  # [C, H, W]
        name = h5["Names"][lid]

        x = torch.from_numpy(img)
        x = self.transform(x)
        return x, decode_name(name)  

BOMI_meanstds = {  
                                'DAPI' : [0.739118, 0.546747],
                                'CD4': [0.392819, 0.663878],
                                'CD20': [0.059158, 0.275279],
                                'CD8': [0.121200, 0.299337],
                                'FoxP3': [0.104232, 0.176990],
                                'PanCK': [0.894238, 1.004463],
                                'Autofluorescence': [1.152428, 0.565964]

                                }


BOMI_CHANNEL_NAMES = [
    "DAPI",
    "CD4",
    "CD20",
    "CD8",
    "FoxP3",
    "PanCK",
    "Autofluorescence",
]




def get_mean_std(args=None):
    channel_names = [
        "DAPI", "CD4", "CD20", "CD8",
        "FoxP3", "PanCK", "Autofluorescence"
    ]

    ds_mean = np.array([BOMI_meanstds[k][0] for k in channel_names], dtype=np.float32)
    ds_std  = np.array([BOMI_meanstds[k][1] for k in channel_names], dtype=np.float32)

    return {"mean": ds_mean, "std": ds_std}, len(channel_names)

# mean_std, nC = get_mean_std()
# print("n_channels:", nC, "mean shape:", mean_std["mean"].shape)
