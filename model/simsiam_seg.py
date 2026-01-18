import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ---- basic building block ----
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


# ---- encoder wrapper that extracts multi-scale features from backbone ----
class SimSiamResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=False)
        # Use the layers up to the final conv block; drop fc, avgpool
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # output stride 4
        self.layer2 = resnet.layer2  # stride 8
        self.layer3 = resnet.layer3  # stride 16
        self.layer4 = resnet.layer4  # stride 32

    def forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))  # /2
        x1 = self.maxpool(x0)                     # /4
        f1 = self.layer1(x1)                      # /4
        f2 = self.layer2(f1)                      # /8
        f3 = self.layer3(f2)                      # /16
        f4 = self.layer4(f3)                      # /32
        return x0, f1, f2, f3, f4


class SimSiam_UNet(nn.Module):
    def __init__(self,
                 num_classes=1,
                 simsiam_ckpt_path="/home/gxu/proj1/lesionSeg/dino_seg/checkpoint/simsiam_res50_epoch100_bs_256.pth.tar",
                 freeze_encoder=True,
                 decoder_channels=(512, 256, 128, 64)):
        super().__init__()
        self.encoder = SimSiamResNetEncoder()

        # load pretrained SimSiam weights if given
        if simsiam_ckpt_path is not None and os.path.exists(simsiam_ckpt_path):
            self._load_simsiam_weights(simsiam_ckpt_path)
        else:
            print("⚠️ No SimSiam checkpoint loaded; encoder randomly initialized.")

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("✅ Encoder frozen. Only decoder will train.")

        # channel dims of encoder
        ch_x0 = 64
        ch_f1 = 256
        ch_f2 = 512
        ch_f3 = 1024
        ch_f4 = 2048

        c_dec4, c_dec3, c_dec2, c_dec1 = decoder_channels

        self.project = nn.Conv2d(ch_f4, c_dec4, kernel_size=1)
        self.dec4 = DecoderBlock(c_dec4, ch_f3, c_dec3)
        self.dec3 = DecoderBlock(c_dec3, ch_f2, c_dec2)
        self.dec2 = DecoderBlock(c_dec2, ch_f1, c_dec1)

        self.up_to_full = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
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
            x0r = F.interpolate(x0, size=d.shape[-2:], mode='bilinear', align_corners=True)
        else:
            x0r = x0

        d = torch.cat([d, x0r], dim=1)
        d = self.reduce_conv(d)
        d = F.interpolate(d, size=(H, W), mode='bilinear', align_corners=True)
        out = self.final_conv(d)
        return out

    def _load_simsiam_weights(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        # many SimSiam checkpoints store their statedict under 'state_dict'
        state = ckpt.get('state_dict', ckpt)

        new_state = {}
        for k, v in state.items():
            # common prefix patterns:
            # module.encoder.*  or encoder.*  or module.model.encoder.* etc
            if k.startswith('module.encoder.') or k.startswith('encoder.'):
                # strip prefix
                if k.startswith('module.encoder.'):
                    new_k = k[len('module.encoder.'):]
                else:
                    new_k = k[len('encoder.'):]
                new_state[new_k] = v
            # maybe projector or predictor parts too, skip them

        enc_dict = self.encoder.state_dict()
        compat = {k: v for k, v in new_state.items() if k in enc_dict and v.shape == enc_dict[k].shape}
        missing = set(enc_dict.keys()) - set(compat.keys())
        print(f"Loaded {len(compat)} tensors into encoder, skipped {len(missing)} missing.")
        enc_dict.update(compat)
        self.encoder.load_state_dict(enc_dict)
        print("✅ SimSiam weights loaded into encoder.")


# ---- Test code ----
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # path to your SimSiam checkpoint
    ckpt = "/home/gxu/proj1/lesionSeg/dino_seg/checkpoint/simsiam_res50_epoch100_bs_256.pth.tar"

    model = SimSiam_UNet(num_classes=4, simsiam_ckpt_path=ckpt, freeze_encoder=True).to(device)

    x = torch.randn(2, 3, 256, 256).to(device)
    with torch.no_grad():
        y = model(x)
    print("Output shape:", y.shape)  # expect (2, 4, 256, 256)
