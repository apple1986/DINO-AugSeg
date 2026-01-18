
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import sys
REPO_DIR = "/home/gxu/proj1/lesionSeg/dino_seg"
sys.path.append(REPO_DIR)


def dinov3_vitb16_mask2form():
    segmentor = torch.hub.load(REPO_DIR, 'dinov3_vitb16_ms', source="local", 
                            backbone_weights="/home/gxu/proj1/lesionSeg/dino_seg/checkpoint/dinov3_vitb16_pretrain_lvd.pth")
    return segmentor

def dinov3_vits16_mask2form():
    segmentor = torch.hub.load(REPO_DIR, 'dinov3_vits16_ms', source="local", 
                            backbone_weights="/home/gxu/proj1/lesionSeg/dino_seg/checkpoint/dinov3_vits16_pretrain_lvd.pth")
    return segmentor


def dinov3_convnext_t_mask2form():
    segmentor = torch.hub.load(REPO_DIR, 'dinov3_convnext_t_ms', source="local", 
                            backbone_weights="/home/gxu/proj1/lesionSeg/dino_seg/checkpoint/dinov3_convnext_tiny_pretrain_lvd.pth")
    return segmentor



"""print layers and params of network"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = dinov3_vitb16_mask2form().to(device)
    model = dinov3_vits16_mask2form().to(device)
    # model = dinov3_convnext_t_mask2form().to(device)
    # torchinfo supports dict outputs
    # summary(model, input_size=(1, 3, 224, 224))

    ## img
    img = torch.randn(1, 3, 512, 512).to(device)
    pd = model.predict(img)
    pd_mask = pd['pred_masks']
    print(torch.unique(pd_mask))
    print(pd.keys())
    print(pd['pred_masks'].shape)
    print(pd['pred_logits'].shape)
