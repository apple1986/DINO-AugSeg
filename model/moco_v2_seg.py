# moco_unet_fixed.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ----------------------------
# Basic conv block used in decoder
# ----------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        # upsample then convs
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = ConvBNReLU(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if skip is not None:
            # match shapes
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# ----------------------------
# ResNet50 encoder wrapper (extract multi-scale features)
# ----------------------------
class ResNet50Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=False)
        self.conv1 = resnet.conv1  # -> /2
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # -> /4
        self.layer1 = resnet.layer1    # -> /4
        self.layer2 = resnet.layer2    # -> /8
        self.layer3 = resnet.layer3    # -> /16
        self.layer4 = resnet.layer4    # -> /32

    def forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))  # /2
        x1 = self.maxpool(x0)                    # /4
        f1 = self.layer1(x1)                     # /4
        f2 = self.layer2(f1)                     # /8
        f3 = self.layer3(f2)                     # /16
        f4 = self.layer4(f3)                     # /32
        return x0, f1, f2, f3, f4


# ----------------------------
# Full UNet-like segmentation model using MoCo-ResNet50 encoder
# ----------------------------
class MoCoResNetUNet(nn.Module):
    def __init__(self,
                 num_classes=1,
                 moco_ckpt_path="/home/gxu/proj1/lesionSeg/dino_seg/checkpoint/moco_v2_800ep_pretrain.pth.tar",
                 freeze_encoder=True,
                 decoder_channels=(512, 256, 128, 64)):
        super().__init__()
        self.encoder = ResNet50Encoder()

        if moco_ckpt_path is not None and os.path.exists(moco_ckpt_path):
            self._load_moco_checkpoint(moco_ckpt_path)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("✅ Encoder frozen (MoCo weights). Only decoder will be trained.")

        ch_x0, ch_f1, ch_f2, ch_f3, ch_f4 = 64, 256, 512, 1024, 2048
        c_dec4, c_dec3, c_dec2, c_dec1 = decoder_channels

        self.project = nn.Conv2d(ch_f4, c_dec4, kernel_size=1)

        self.dec4 = DecoderBlock(c_dec4, ch_f3, c_dec3)
        self.dec3 = DecoderBlock(c_dec3, ch_f2, c_dec2)
        self.dec2 = DecoderBlock(c_dec2, ch_f1, c_dec1)

        self.up_to_full = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # ✅ define reduce_conv ONCE, not inside forward
        self.reduce_conv = ConvBNReLU(c_dec1 + ch_x0, c_dec1)
        self.final_conv = nn.Sequential(
            ConvBNReLU(c_dec1, c_dec1 // 2),
            nn.Conv2d(c_dec1 // 2, num_classes, kernel_size=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x0, f1, f2, f3, f4 = self.encoder(x)

        d = self.project(f4)
        d = self.dec4(d, f3)
        d = self.dec3(d, f2)
        d = self.dec2(d, f1)

        d = self.up_to_full(d)
        if x0.shape[-2:] != d.shape[-2:]:
            x0 = F.interpolate(x0, size=d.shape[-2:], mode='bilinear', align_corners=True)

        d = torch.cat([d, x0], dim=1)
        d = self.reduce_conv(d)
        d = F.interpolate(d, size=(H, W), mode='bilinear', align_corners=True)
        out = self.final_conv(d)
        return out

    def _load_moco_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)
        new_state = {}
        for k, v in state_dict.items():
            if k.startswith('module.encoder_q.'):
                new_k = k[len('module.encoder_q.'):]
            elif k.startswith('encoder_q.'):
                new_k = k[len('encoder_q.'):]
            else:
                continue
            new_state[new_k] = v

        encoder_dict = self.encoder.state_dict()
        compatible = {k: v for k, v in new_state.items()
                      if k in encoder_dict and v.shape == encoder_dict[k].shape}
        encoder_dict.update(compatible)
        self.encoder.load_state_dict(encoder_dict)
        print(f"✅ Loaded MoCo weights ({len(compatible)} tensors matched).")


# ----------------------------
# Test: works on CPU and CUDA
# ----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MoCoResNetUNet(
        num_classes=3,
        moco_ckpt_path="/home/gxu/proj1/lesionSeg/dino_seg/checkpoint/moco_v2_800ep_pretrain.pth.tar",
        freeze_encoder=True
    ).to(device)

    x = torch.randn(2, 3, 256, 256).to(device)
    model.eval()
    with torch.no_grad():
        y = model(x)
    print("✅ Output shape:", y.shape)
