

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


class Synapse(Dataset):
    def __init__(self, root, split='trainval', transform_img=None, img_size=768, train_num="all"):
        self.root = root
        self.split = split
        self.transform_img = transform_img
        self.img_size = img_size
        self.split = split

        if self.split == 'train':
            list_file = os.path.join(self.root, "dataset/Synapse/train_dino.txt")
            data_dir  = os.path.join(self.root, "dataset/Synapse/train_val_nii")

        elif self.split == 'val':
            list_file = os.path.join(self.root, "dataset/Synapse/val_dino.txt")
            data_dir  = os.path.join(self.root, "dataset/Synapse/train_val_nii")

        elif self.split == 'trainval':
            # combine train + val lists
            list_file = os.path.join(self.root, "dataset/Synapse/trainval_dino.txt")
            data_dir  = os.path.join(self.root, "dataset/Synapse/train_val_nii")

        elif self.split == 'test':
            list_file = os.path.join(self.root, "dataset/Synapse/test_dino.txt")
            data_dir  = os.path.join(self.root, "dataset/Synapse/test_nii")

        else:
            raise ValueError("split must be 'train'|'val'|'trainval'|'test'")


        with open(list_file, 'r') as f:
            lines = f.readlines()
        # build full paths WITHOUT extension, keep only prefix
        self.img_paths = [
            os.path.join(data_dir, line.strip()) for line in lines
        ]
        # set the number of training samples
        if (self.split == "train") and (train_num != "all"):
            self.img_paths = self.img_paths[:train_num]


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = nib.load(self.img_paths[idx] + "_ct.nii.gz").get_fdata()  # (H, W, D)
        mask = nib.load(self.img_paths[idx] + "_gt.nii.gz").get_fdata()  # (H, W, D)
        _, _, D = img.shape
        # if self.split == "val":
        img_vol = torch.zeros(3, self.img_size, self.img_size, D) # HxWxD
        msk_vol = torch.zeros(self.img_size, self.img_size, D, dtype=torch.long) # HxWxD
        for idx in range(img.shape[2]):
            img_one = img[:, :, idx]
            msk_one = mask[:, :, idx]
            # normalize image
            img_one = (img_one - img_one.min()) / (img_one.max() - img_one.min() + 1e-8)
            img_one = np.stack([img_one]*3, axis=-1)  # make 3-channel
            # apply image transform
            if self.transform_img:
                img_one = self.transform_img(Image.fromarray((img_one*255).astype(np.uint8))) # 3xHxW
                img_vol[:, :, :, idx] = img_one
                # resize mask separately (nearest interp, keep ints)
                msk_one = Image.fromarray(msk_one.astype(np.uint8))
                msk_one = msk_one.resize((self.img_size, self.img_size), resample=Image.NEAREST)
                msk_one = torch.from_numpy(np.array(msk_one)).long()  # [H,W], values {0,1,2,3}   
                msk_vol[:, :, idx] = msk_one
            else:
                img_one = torch.tensor(img_one.transpose(2,0,1), dtype=torch.float32)
                img_vol[:, :, :, idx] = img_one
                # resize mask separately (nearest interp, keep ints)
                msk_one = Image.fromarray(msk_one.astype(np.uint8))
                msk_one = msk_one.resize((self.img_size, self.img_size), resample=Image.NEAREST)
                msk_one = torch.from_numpy(np.array(msk_one)).long()  # [H,W], values {0,1,2,3}
                msk_vol[:, :, idx] = msk_one
        return img_vol, msk_vol # 3HWD and HWD

        # else: # training data
        #     # 这里假设按 slice 展开
        #     slice_idx = np.random.randint(img.shape[-1])
        #     img = img[:, :, slice_idx]
        #     mask = mask[:, :, slice_idx]

        #     # normalize image
        #     img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        #     img = np.stack([img]*3, axis=-1)  # make 3-channel

        #     # apply image transform
        #     if self.transform_img:
        #         img = self.transform_img(Image.fromarray((img*255).astype(np.uint8)))
        #         # resize mask separately (nearest interp, keep ints)
        #         mask = Image.fromarray(mask.astype(np.uint8))
        #         mask = mask.resize((self.img_size, self.img_size), resample=Image.NEAREST)
        #         mask = torch.from_numpy(np.array(mask)).long()  # [H,W], values {0,1,2,3}            
        #     else:
        #         img = torch.tensor(img.transpose(2,0,1), dtype=torch.float32)
        #         # resize mask separately (nearest interp, keep ints)
        #         mask = Image.fromarray(mask.astype(np.uint8))
        #         mask = mask.resize((self.img_size, self.img_size), resample=Image.NEAREST)
        #         mask = torch.from_numpy(np.array(mask)).long()  # [H,W], values {0,1,2,3}

        #     return img, mask