import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Add MAE repo path ----
mae_repo_path = "/home/gxu/proj1/lesionSeg/github/mae"
if mae_repo_path not in sys.path:
    sys.path.insert(0, mae_repo_path)

# Import both models from the MAE repo
from models_mae import mae_vit_base_patch16, mae_vit_large_patch16


class MAE_Segmentation(nn.Module):
    def __init__(self,
                 model_name="mae_vit_large_patch16",
                 mask_ratio=0.0,
                 num_classes=1,
                 img_size=224,
                 patch_size=16,
                 decoder_channels=[512, 256, 128, 64],
                 pretrained_path="/home/gxu/proj1/lesionSeg/dino_seg/checkpoint/mae_pretrain_vit_large.pth",
                 freeze_encoder=True):
        super().__init__()

        # ---- Select and instantiate MAE backbone ----
        if model_name == "mae_vit_base_patch16":
            self.mae = mae_vit_base_patch16(img_size=img_size, in_chans=3)
            self.embed_dim = 768
        elif model_name == "mae_vit_large_patch16":
            self.mae = mae_vit_large_patch16(img_size=img_size, in_chans=3)
            self.embed_dim = 1024
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        # ---- Optionally load pretrained weights ----
        if pretrained_path is not None and os.path.exists(pretrained_path):
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            state_dict = checkpoint.get('model', checkpoint)
            msg = self.mae.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded pretrained MAE weights from {pretrained_path}")
        else:
            print("⚠️ No pretrained weights loaded (using random initialization).")

        # ---- Optionally freeze encoder ----
        if freeze_encoder:
            for param in self.mae.parameters():
                param.requires_grad = False
            print("✅ Encoder frozen. Only decoder will be trained.")

        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        # ---- Lightweight UNet-style decoder ----
        self.proj = nn.Conv2d(self.embed_dim, decoder_channels[0], kernel_size=1)
        self.up1 = nn.ConvTranspose2d(decoder_channels[0], decoder_channels[1], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(decoder_channels[1], decoder_channels[2], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(decoder_channels[2], decoder_channels[3], kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(decoder_channels[3], num_classes, kernel_size=1)

    def forward(self, x):
        """
        x: (B, 3, H, W)
        returns: segmentation logits (B, num_classes, H, W)
        """
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"Image size must be multiple of patch size {self.patch_size}"

        # ---- Forward encoder (no masking for segmentation) ----
        latent, _, _ = self.mae.forward_encoder(x, mask_ratio=self.mask_ratio)
        # latent: [B, N+1, embed_dim], includes cls token
        latent = latent[:, 1:, :]  # remove cls token

        # ---- Reshape patch tokens to feature map ----
        H_patch, W_patch = H // self.patch_size, W // self.patch_size
        x_patch = latent.transpose(1, 2).contiguous().view(B, self.embed_dim, H_patch, W_patch)

        # ---- Decode ----
        x = self.proj(x_patch)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        logits = self.final_conv(x)
        return logits


# ---- Example usage ----
if __name__ == "__main__":
    model = MAE_Segmentation(
        model_name="mae_vit_large_patch16",  # or "mae_vit_large_patch16"
        pretrained_path="/home/gxu/proj1/lesionSeg/dino_seg/checkpoint/mae_pretrain_vit_large.pth",
        mask_ratio=0.0,  # no masking for segmentation
        num_classes=4,
        img_size=224,
        patch_size=16,
        decoder_channels=[512, 256, 128, 64],
        freeze_encoder=True,  # only train decoder
    )

    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # expected: (2, 4, 224, 224)
