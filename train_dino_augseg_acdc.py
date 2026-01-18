
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import os
from torchvision import transforms

from builders.model_builder import build_model
from dataset.acdc import ACDCDataset
from model.Dinov3_WTAUG_UNet import DINO_AugSeg


# ------------------------------
# 2. Loss function (BCE + Dice)
# ------------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

def bce_dice_loss(pred, target):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = DiceLoss()(pred, target)
    return bce + dice

class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes=4, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, preds, targets):
        """
        preds: [B,C,H,W] (logits)
        targets: [B,H,W] (0..C-1)
        """
        preds = torch.softmax(preds, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes)  # [B,H,W,C]
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()       # [B,C,H,W]

        dims = (0,2,3)  # sum over batch+spatial dims
        intersection = torch.sum(preds * targets_onehot, dims)
        cardinality = torch.sum(preds + targets_onehot, dims)

        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice.mean()   # average over classes

class CombinedLoss(nn.Module):
    def __init__(self, num_classes=4, alpha=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = MultiClassDiceLoss(num_classes=num_classes)
        self.alpha = alpha  # weight for CE

    def forward(self, preds, targets):
        ce_loss = self.ce(preds, targets)
        dice_loss = self.dice(preds, targets)
        return self.alpha * ce_loss + (1 - self.alpha) * dice_loss

def dice_score_per_class(preds, targets, num_classes=4, smooth=1e-5):
    """
    preds: [B,C,H,W] (logits)
    targets: [B,H,W]
    """
    preds = torch.softmax(preds, dim=1).argmax(dim=1)  # [B,H,W]

    dice_scores = []
    for c in range(num_classes):
        pred_c = (preds == c).float()
        target_c = (targets == c).float()

        intersection = (pred_c * target_c).sum()
        denom = pred_c.sum() + target_c.sum()
        dice = (2 * intersection + smooth) / (denom + smooth)
        dice_scores.append(dice.item())

    return dice_scores  # list of length num_classes
# ------------------------------
# 3. Training & Validation Loop
# ------------------------------
def train_model(
    model,
    train_loader,
    device,
    epochs=20,
    lr=1e-4,
    alpha=0.5,
    save_model_path=None
):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = CombinedLoss(num_classes=4, alpha=alpha)

    os.makedirs(save_model_path, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}"
        )

        # optional: log training loss
        if save_model_path is not None:
            with open(os.path.join(save_model_path, "train_loss.txt"), "a") as f:
                f.write(
                    f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}\n"
                )

    # ---- save last epoch model only ----
    if save_model_path is not None:
        last_ckpt = os.path.join(save_model_path, "acdc_model_last_epoch.pth")
        torch.save(model.state_dict(), last_ckpt)
        print(f"Saved last epoch model to: {last_ckpt}")

    return model


# image transformations
def img_transform_torch(image_size):
    img_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),   # resize image
        transforms.ToTensor(),           # [3,H,W] float
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),  # ImageNet stats
            std=(0.229, 0.224, 0.225),
        )
    ])
    return img_transform
##############################################################################
# Main function to train DINOv3 ConvNeXt U-Net on ACDC
##############################################################################

# set hyperparmeters
root_path = "/home/gxu/proj1/lesionSeg"
REPO_DIR = os.path.join(root_path, "dino_augseg")
# "segdino", "unet", "segnet", "unetpp", "attunet", "multiresunet",    "r2attunet", "r2unet", 
MODEL_NAMES = ["cross_guide_wt_unet",] #  mae_seg simsiam_seg mocov2_seg aim_seg simsiam_seg
# MODEL_NAMES = ["cross_guide_wt_unet", "attunet", "missformer",  "multiresunet",  "nnUNet",
#                "segdino",  "segformer", "segnet", "SwinUNETR", "unet", "unetpp"] #  
image_size = 768

# training
USE_WT_AUG = True # whether to do feature augmentation on wavelet dimension
USE_AUG_FEAT = False  # whether to do feature augmentation on spatial dimension
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for decoder_name in MODEL_NAMES: 
    for num in [7]: # 2, 1, "all", 
        print(f"training samples: {num}")
        print(f"decoder_name: {decoder_name}")
        save_model_path = os.path.join(REPO_DIR, "checkpoint/checkpoint_acdc/dino_augseg", decoder_name+"_"+str(num))
        # make dir if not exist
        os.makedirs(save_model_path, exist_ok=True)
        # dataset
        train_dataset = ACDCDataset(REPO_DIR, split="trainval", transform_img=img_transform_torch(image_size), train_num=num, img_size=image_size)
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        # build model 
        if decoder_name == "segdino":
            model_weight_path = os.path.join(root_path, "dino_augseg/checkpoint/dino_ori/dinov3_vitb16_pretrain_lvd.pth") 
            backbone = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights=model_weight_path)
            model = build_model(decoder_name, num_classes=4, backbone=backbone) 
        elif decoder_name in ["cross_guide_wt_unet",]:
            # build model (pick U-Net style or FCN-style)
            # DINOv3 ConvNeXt models pretrained on web images
            model_weight_path = os.path.join(root_path, "dino_augseg/checkpoint/dino_ori/dinov3_convnext_large_pretrain_lvd.pth")
            dinov3_convnext_tiny = torch.hub.load(REPO_DIR, 'dinov3_convnext_large', source='local', weights=model_weight_path)
            model = DINO_AugSeg(dinov3_convnext_tiny, num_classes=4, model_type="large", decoder_type=decoder_name, 
                                use_wt_aug=USE_WT_AUG, aug_feat=USE_AUG_FEAT)                    
        else:
            model = build_model(decoder_name, num_classes=4)
        # train the model
        trained_model = train_model(model, train_loader, device=device, epochs=2000, lr=1e-4, save_model_path=save_model_path)
