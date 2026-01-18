

import numpy as np
import os
from PIL import Image

from scipy.ndimage.interpolation import zoom
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ACDCDataset(Dataset):
    def __init__(self, root, split='trainval', transform_img=None, img_size=768, train_num="all"):
        self.root = root
        self.split = split
        self.transform_img = transform_img
        self.img_size = img_size

        if self.split == 'train':
            list_file = os.path.join(self.root, "dataset/ACDC/train.list")
            data_dir  = os.path.join(self.root, "dataset/ACDC/train_data")
        elif self.split == 'val':
            list_file = os.path.join(self.root, "dataset/ACDC/val.list")
            data_dir  = os.path.join(self.root, "dataset/ACDC/val_data")
        elif self.split == "trainval":
            # combine train + val
            list_file = os.path.join(self.root, "dataset/ACDC/trainval.list")
            data_dir  = os.path.join(self.root, "dataset/ACDC/trainval_data")

        elif self.split == 'test':
            list_file = os.path.join(self.root, "dataset/ACDC/test.list")
            data_dir  = os.path.join(self.root, "dataset/ACDC/test_data")
        else:
            raise ValueError("split must be 'train'|'val'|'trainval'|'test'")

        with open(list_file, 'r') as f:
            lines = f.readlines()
        # build full paths WITHOUT extension, keep only prefix
        self.img_paths = [
            os.path.join(data_dir, line.strip()) for line in lines
        ]
        # set the number of training samples
        if (self.split == "trainval") and (train_num != "all"):
            self.img_paths = self.img_paths[:train_num]


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = nib.load(self.img_paths[idx] + "_mr.nii.gz").get_fdata()  # (H, W, D)
        mask = nib.load(self.img_paths[idx] + "_gt.nii.gz").get_fdata()  # (H, W, D)

        # slect slice
        slice_idx = np.random.randint(img.shape[-1])
        img = img[:, :, slice_idx]
        mask = mask[:, :, slice_idx]

        # normalize image
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = np.stack([img]*3, axis=-1)  # make 3-channel

        # apply image transform
        if self.transform_img:
            img = self.transform_img(Image.fromarray((img*255).astype(np.uint8)))
        else:
            img = torch.tensor(img.transpose(2,0,1), dtype=torch.float32)

        # resize mask separately (nearest interp, keep ints)
        mask = Image.fromarray(mask.astype(np.uint8))
        mask = mask.resize((self.img_size, self.img_size), resample=Image.NEAREST)
        mask = torch.from_numpy(np.array(mask)).long()  # [H,W], values {0,1,2,3}

        return img, mask


# apply Albumentations
class ACDCDatasetV2(Dataset):
    def __init__(self, root, split='train', transform_img=None, img_size=768, train_num="all"):
        self.root = root
        self.split = split
        self.transform = transform_img
        self.img_size = img_size

        if self.split == 'train':
            list_file = os.path.join(self.root, "dataset/ACDC/train.list")
            data_dir  = os.path.join(self.root, "dataset/ACDC/train_data")
        elif self.split == 'val':
            list_file = os.path.join(self.root, "dataset/ACDC/val.list")
            data_dir  = os.path.join(self.root, "dataset/ACDC/val_data")
        elif self.split == 'test':
            list_file = os.path.join(self.root, "dataset/ACDC/test.list")
            data_dir  = os.path.join(self.root, "dataset/ACDC/test_data")
        else:
            raise ValueError("split must be 'train'|'val'|'test'")

        with open(list_file, 'r') as f:
            lines = f.readlines()
        self.img_paths = [os.path.join(data_dir, line.strip()) for line in lines]

        if (self.split == "train") and (train_num != "all"):
            self.img_paths = self.img_paths[:train_num]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = nib.load(self.img_paths[idx] + "_mr.nii.gz").get_fdata()
        mask = nib.load(self.img_paths[idx] + "_gt.nii.gz").get_fdata()

        slice_idx = np.random.randint(img.shape[-1])
        img = img[:, :, slice_idx]
        mask = mask[:, :, slice_idx]

        # normalize image to [0,1] and convert to 3 channels
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = np.stack([img]*3, axis=-1).astype(np.float32)
        mask = mask.astype(np.int64)

        # apply Albumentations
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        else:
            # fallback: convert to torch tensor
            img = torch.from_numpy(img.transpose(2,0,1))
            mask = torch.from_numpy(mask)

        return img, mask.long()
    
# apply Albumentations
class ACDCDataset_Aug(Dataset):
    def __init__(self, root, split='train', transform_img=None, img_size=768, train_num="all"):
        self.root = root
        self.split = split
        self.transform = transform_img
        self.img_size = img_size

        if self.split == 'train':
            list_file = os.path.join(self.root, "dataset/ACDC/train.list")
            data_dir  = os.path.join(self.root, "dataset/ACDC/train_data")
        elif self.split == 'val':
            list_file = os.path.join(self.root, "dataset/ACDC/val.list")
            data_dir  = os.path.join(self.root, "dataset/ACDC/val_data")
        elif self.split == 'test':
            list_file = os.path.join(self.root, "dataset/ACDC/test.list")
            data_dir  = os.path.join(self.root, "dataset/ACDC/test_data")
        else:
            raise ValueError("split must be 'train'|'val'|'test'")

        with open(list_file, 'r') as f:
            lines = f.readlines()

        # full paths WITHOUT extension
        self.img_paths = [os.path.join(data_dir, line.strip()) for line in lines]

        if (self.split == "train") and (train_num != "all"):
            self.img_paths = self.img_paths[:train_num]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = nib.load(self.img_paths[idx] + "_mr.nii.gz").get_fdata()  # (H,W,D)
        mask = nib.load(self.img_paths[idx] + "_gt.nii.gz").get_fdata()  # (H,W,D)

        # pick one slice
        slice_idx = np.random.randint(img.shape[-1])
        img = img[:, :, slice_idx]
        mask = mask[:, :, slice_idx]

        # normalize image to [0,1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # make 3-channel
        img = np.stack([img]*3, axis=-1).astype(np.float32)  # [H,W,3]
        mask = mask.astype(np.uint8)                        # [H,W]

        # Albumentations expects numpy arrays
        if self.transform:
            img, mask = self.transform(img, mask)
        else:
            # fallback: manual tensor conversion
            img = torch.tensor(img.transpose(2,0,1), dtype=torch.float32)
  
        # resize mask separately (nearest interp, keep ints)
        mask = Image.fromarray(mask.astype(np.uint8))
        mask = mask.resize((self.img_size, self.img_size), resample=Image.NEAREST)
        mask = torch.from_numpy(np.array(mask)).long()  # [H,W], values {0,1,2,3}

        return img, mask