import torch
import torch.nn as nn
import torch.nn.functional as F

# import AIM loader
from aim.v2.utils import load_pretrained


# --------------------------------
# Conv block for decoder
# --------------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = ConvBNReLU(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if skip is not None:
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# --------------------------------
# AIMv2 + UNet segmentation model
# --------------------------------
class AIMv2_UNet(nn.Module):
    def __init__(self,
                 model_id="aimv2-large-patch14-224",
                 num_classes=1,
                 freeze_encoder=True,
                 decoder_channels=(512, 256, 128, 64)):
        super().__init__()

        # load pretrained AIMv2 model
        self.aim = load_pretrained(model_id, backend="torch")
        print("Loaded AIMv2 model:", model_id)

        # Optionally freeze encoder
        if freeze_encoder:
            for p in self.aim.parameters():
                p.requires_grad = False
            print("Encoder frozen. Only decoder will train.")

        # Determine patch size and embedding dimension from the model
        # (You will need to inspect the AIM model for attributes like patch_size or embed_dim)
        self.patch_size = self.aim.patch_size if hasattr(self.aim, 'patch_size') else 14
        # determine AIMv2 encoder embedding dimension
        # determine AIMv2 encoder embedding dimension
        if hasattr(self.aim, 'embed_dim'):
            self.embed_dim = self.aim.embed_dim
        elif hasattr(self.aim, 'hidden_dim'):
            self.embed_dim = self.aim.hidden_dim
        elif hasattr(self.aim, 'preprocessor') and hasattr(self.aim.preprocessor, 'patchifier') and hasattr(self.aim.preprocessor.patchifier, 'proj'):
            self.embed_dim = self.aim.preprocessor.patchifier.proj.out_channels
        else:
            raise ValueError("Cannot determine AIMv2 encoder embedding dimension")


        c_dec4, c_dec3, c_dec2, c_dec1 = decoder_channels
        self.proj = nn.Conv2d(self.embed_dim, c_dec4, kernel_size=1)

        # If AIM supports multi-scale features, you'd define skip layers here.
        # For now, we assume single feature and no skip features.
        self.dec4 = DecoderBlock(c_dec4, skip_ch=0, out_ch=c_dec3)
        self.dec3 = DecoderBlock(c_dec3, skip_ch=0, out_ch=c_dec2)
        self.dec2 = DecoderBlock(c_dec2, skip_ch=0, out_ch=c_dec1)

        self.up_to_full = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Sequential(
            ConvBNReLU(c_dec1, c_dec1 // 2),
            nn.Conv2d(c_dec1 // 2, num_classes, kernel_size=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"Image size must be divisible by patch size {self.patch_size}"

        # forward through AIMv2 (encoder trunk)
        # The `load_pretrained` model likely provides a forward that returns image features
        # For many AIM models: features, _ = model.forward(x) or model(x)
        feats = self.aim(x)  # shape: (B, num_patches, embed_dim) or similar  
        # If it returns logits as well, extract just features.

        # # assuming feats is [B, N, embed_dim]
        # # drop class token if present
        # if feats.dim() == 3:
        #     N = feats.shape[1]
        #     # if there's a cls token, drop it (first token)
        #     # this is heuristic — you may confirm via model architecture
        #     feats = feats[:, 1:, :]

        # reshape to 2D feature map
        H_patch = H // self.patch_size
        W_patch = W // self.patch_size
        feat_map = feats.transpose(1, 2).contiguous().view(B, self.embed_dim, H_patch, W_patch)

        # decode
        x_d = self.proj(feat_map)
        x_d = self.dec4(x_d, skip=None)
        x_d = self.dec3(x_d, skip=None)
        x_d = self.dec2(x_d, skip=None)
        x_d = self.up_to_full(x_d)
        x_d = F.interpolate(x_d, size=(H, W), mode='bilinear', align_corners=True)
        out = self.final_conv(x_d)
        return out


# --------------------------------
# Test usage
# --------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AIMv2_UNet(
        model_id="aimv2-large-patch14-224",
        num_classes=3,
        freeze_encoder=True
    ).to(device)

    inp = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        out = model(inp)
    print("Output shape:", out.shape)  # expected e.g. (2, 3, 224, 224)
