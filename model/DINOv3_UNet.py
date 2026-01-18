import torch
import torch.nn as nn
import torch.nn.functional as F
    

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2=None):
        if x2 is not None:
            diffY = x1.size()[2] - x2.size()[2]
            diffX = x1.size()[3] - x2.size()[3]
            x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
            x = torch.cat([x1, x2], dim=1)
        else:
            x = x1
        x = self.up(x)
        return self.conv(x)


class DINOv3_UNet(nn.Module):
    def __init__(self, dinov3_weight_path=None, dinov3_local_path="./", model_type="dinov3_vitl16"):
        super(DINOv3_UNet, self).__init__()
        '''
        All models: 
            dinov3_convnext_base
            dinov3_convnext_large
            dinov3_convnext_small
            dinov3_convnext_tiny
            dinov3_vit7b16
            dinov3_vitb16
            dinov3_vith16plus
            dinov3_vitl16 <=
            dinov3_vitl16plus
            dinov3_vits16
            dinov3_vits16plus
        '''
        self.dino = torch.hub.load(
            repo_or_dir=dinov3_local_path,
            model=model_type,
            source="local",
            pretrained=False,
            trust_repo=True
        )
        if dinov3_weight_path:
            checkpoint = torch.load(dinov3_weight_path, map_location='cpu')
            self.dino.load_state_dict(checkpoint, strict=True)
            print("✓ Local weights successfully loaded")

        for param in self.dino.parameters():
            param.requires_grad = False

        self.reduce1 = nn.Conv2d(1024, 128, 1)
        self.reduce2 = nn.Conv2d(1024, 128, 1)
        self.reduce3 = nn.Conv2d(1024, 128, 1)
        self.reduce4 = nn.Conv2d(1024, 128, 1)

        self.up1 = Up(256, 128)
        self.up2 = Up(256, 128)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 128)
        self.head = nn.Conv2d(128, 1, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.dino.forward_features(x)['x_norm_patchtokens']
        x = x.view(B, H//16, W//16, -1).permute(0, 3, 1, 2)
        x1 = F.interpolate(self.reduce1(x), size=(H//4, W//4), mode='bilinear')
        x2 = F.interpolate(self.reduce2(x), size=(H//8, W//8), mode='bilinear')
        x3 = F.interpolate(self.reduce3(x), size=(H//16, W//16), mode='bilinear')
        x4 = F.interpolate(self.reduce4(x), size=(H//32, W//32), mode='bilinear')
        x = self.up4(x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        out = self.head(x)
        out = F.interpolate(self.head(x), scale_factor=2, mode='bilinear')
        return out
    

if __name__ == "__main__":
    dinov3_local_path = "/home/gxu/proj1/lesionSeg/dino_seg"
    dinov3_weight_path = "/home/gxu/proj1/lesionSeg/dino_seg/checkpoint/dinov3_vitl16_pretrain_lvd.pth"
    model_type = "dinov3_vitl16"
    model = DINOv3_UNet(dinov3_local_path=dinov3_local_path, 
                        dinov3_weight_path=dinov3_weight_path, model_type=model_type).cuda().eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 448, 448).cuda()
        out = model(x)
        print(out.shape)