import torch
import torch.nn as nn
import os

def load_dinov3_encoder(model_type="convnext_tiny", repo_dir=None, device=None):
    """
    Load a pretrained DINOv3 model as encoder.
    
    Args:
        model_type (str): e.g., 'convnext_tiny', 'convnext_small', 'vit_b', etc.
        repo_dir (str): local path to DINO repo (for torch.hub.load).
        weight_path (str): path to pretrained weights (.pth). If None, uses default weights.
        device (torch.device): device to move the encoder to (default: cuda if available).
    
    Returns:
        encoder (nn.Module): DINOv3 model with head replaced by Identity.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = f"dinov3_{model_type}"
    weight_path = os.path.join(repo_dir, "checkpoint/"+model_type+"_pretrain_lvd.pth")
    
    # Load pretrained model using torch.hub from local repo
    if weight_path is not None:
        encoder = torch.hub.load(repo_dir, model_type, source='local', weights=weight_path)
    else:
        encoder = torch.hub.load(repo_dir, model_type, source='local', pretrained=True)
    
    # Remove classifier head
    encoder.head = nn.Identity()
    
    # Move to device
    encoder = encoder.to(device)
    
    # Freeze encoder weights by default
    for p in encoder.parameters():
        p.requires_grad = False
    
    return encoder


# extract features from DINOv3 encoder
def extract_encoder_features(encoder, img_tensor, model_type="convnext"):
    """
    Extract multi-stage features from DINOv3 encoder.
    
    Args:
        encoder: pretrained DINOv3 encoder (ConvNeXt or ViT)
        img_tensor: input image tensor [B, 3, H, W]
        model_type: 'convnext' or 'vit'
    
    Returns:
        List of 4 feature maps [f1, f2, f3, f4], each [B, C, H, W]
    """
    encoder.eval()
    features = []
    with torch.no_grad():
        if model_type.lower() == "convnext":
            out = img_tensor
            for i, down in enumerate(encoder.downsample_layers):
                out = down(out)
                out = encoder.stages[i](out)
                features.append(out)
    return features

## build a U_net like model using DINOv3 encoder and test





if __name__ == "__main__":
    REPO_DIR = "/home/gxu/proj1/lesionSeg/dino_seg"
    MODEL_TYPE = "convnext_tiny"
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encoder = load_dinov3_encoder(model_type=MODEL_TYPE, repo_dir=REPO_DIR, device=device)

    # Test forward
    img_tensor = torch.randn((1, 3, 224, 224)).to(device)
    # Extract features
    features = extract_encoder_features(encoder, img_tensor, model_type="convnext")

    # Print feature shapes
    for i, f in enumerate(features):
        print(f"Stage {i}: shape {f.shape}")