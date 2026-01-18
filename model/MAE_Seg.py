import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Add MAE repo path ----
mae_repo_path = "/home/gxu/proj1/lesionSeg/github/mae"
if mae_repo_path not in sys.path:
    sys.path.insert(0, mae_repo_path)

from models_mae import mae_vit_base_patch16, mae_vit_large_patch16


class MAE_Segmentation(nn.Module):
    def __init__(self,
                 model_name="mae_vit_large_patch16",
                 pretrained_path="/home/gxu/proj1/lesionSeg/dino_seg/checkpoint/mae_pretrain_vit_large.pth",
                 mask_ratio=0.0,
                 num_classes=1,
                 img_size=224,
                 patch_size=16,
                 decoder_channels=[512, 256, 128, 64],
                 freeze_mae=True):
        super().__init__()

        # ---- Load MAE backbone ----
        if model_name == "mae_vit_base_patch16":
            self.mae = mae_vit_base_patch16(img_size=img_size, in_chans=3)
            self.embed_dim = 768
            self.decoder_dim = 512
        elif model_name == "mae_vit_large_patch16":
            self.mae = mae_vit_large_patch16(img_size=img_size, in_chans=3)
            self.embed_dim = 1024
            self.decoder_dim = 768
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # ---- Optionally load pretrained weights ----
        if pretrained_path is not None and os.path.exists(pretrained_path):
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            state_dict = checkpoint.get('model', checkpoint)
            msg = self.mae.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded pretrained weights from {pretrained_path}")
            if len(msg.missing_keys) > 0 or len(msg.unexpected_keys) > 0:
                print(f"⚠️ Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")
        else:
            print("⚠️ No pretrained weights loaded (using random initialization).")

        # ---- Freeze encoder and decoder if desired ----
        if freeze_mae:
            for param in self.mae.parameters():
                param.requires_grad = False
            print("✅ MAE encoder+decoder frozen. Only UNet head will be trained.")

        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

        # ---- UNet-like decoder ----
        self.proj = nn.Conv2d(self.decoder_dim, decoder_channels[0], kernel_size=1)
        self.up1 = nn.ConvTranspose2d(decoder_channels[0], decoder_channels[1], 2, stride=2)
        self.up2 = nn.ConvTranspose2d(decoder_channels[1], decoder_channels[2], 2, stride=2)
        self.up3 = nn.ConvTranspose2d(decoder_channels[2], decoder_channels[3], 2, stride=2)
        self.final_conv = nn.Conv2d(decoder_channels[3], num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass: use MAE encoder + decoder as feature extractor
        and feed decoder output to UNet-style convolutional decoder.
        """
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0

        # ---- MAE forward pass ----
        latent, _, ids_restore = self.mae.forward_encoder(x, mask_ratio=self.mask_ratio)
        decoded = self.mae.forward_decoder(latent, ids_restore)  # [B, num_patches, decoder_dim]

        # ---- Convert decoder output to spatial map ----
        # decoded = decoded[:, 1:, :]  # drop cls token
        H_patch, W_patch = H // self.patch_size, W // self.patch_size
        feat_map = decoded.transpose(1, 2).contiguous().view(B, self.decoder_dim, H_patch, W_patch)

        # ---- UNet-like decoder ----
        x = self.proj(feat_map)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        logits = self.final_conv(x)

        return logits


# ---- Example Usage ----
if __name__ == "__main__":
    model = MAE_Segmentation(
        model_name="mae_vit_large_patch16",
        pretrained_path="/home/gxu/proj1/lesionSeg/dino_seg/checkpoint/mae_pretrain_vit_large.pth",
        freeze_mae=True,
        num_classes=4,
        img_size=224,
        patch_size=16,
    )

    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print("Output shape:", out.shape)
