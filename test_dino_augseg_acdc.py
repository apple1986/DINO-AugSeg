import os
import torch
import numpy as np
from torchvision import transforms

from utils.eval_model import test_model
from model.Dinov3_WTAUG_UNet import DINO_AugSeg
from builders.model_builder import build_model


#######################
## testing
# load trained model
root_path = "/home/gxu/proj1/lesionSeg"
REPO_DIR = os.path.join(root_path, "dino_augseg")
DECODER = ["cross_guide_wt_unet", ] 
img_size = 768
# test all trained models
split = "test"
use_wt_aug = False  # False  # whether to use wt aug in testing
aug_feat = False  # False  # whether to use aug feat in testing
for decoder_name in DECODER:
    print(f"model: {decoder_name}")
    for train_num in [ 7, ]: # 1, 2, "all",
        print(train_num)
        MODEL_NAMES = "checkpoint_acdc/dino_augseg/"+decoder_name+"_"+str(train_num)
        save_model_path = os.path.join(REPO_DIR, "checkpoint", MODEL_NAMES)
        # load pretrained convnext_tiny from your repo (head=Identity)  
        # build model 
        if decoder_name in ["cross_guide_wt_unet",]:
            # build model (pick U-Net style or FCN-style)
            # DINOv3 ConvNeXt models pretrained on web images
            model_weight_path = os.path.join(root_path, "checkpoint/dino_ori/dinov3_convnext_large_pretrain_lvd.pth")
            dinov3_convnext_tiny = torch.hub.load(REPO_DIR, 'dinov3_convnext_large', source='local', weights=model_weight_path)
            model = DINO_AugSeg(dinov3_convnext_tiny, num_classes=4, model_type="large", decoder_type=decoder_name, use_wt_aug=use_wt_aug, aug_feat=aug_feat)             
        
        else:
            model = build_model(decoder_name, num_classes=4)
        model.load_state_dict(torch.load(os.path.join(save_model_path, "acdc_model_last_epoch.pth"), map_location="cuda"))

        img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        ])
        # run test: test_list="dataset/ACDC/test.list",
        results = test_model(model, img_transform=img_transform,
                            root=REPO_DIR, save_file=os.path.join(save_model_path, "acdc_results_"+split+".txt"),
                            img_size=img_size, num_classes=4, device="cuda", split=split)
        
        