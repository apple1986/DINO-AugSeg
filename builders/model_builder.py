from model.SQNet import SQNet
from model.LinkNet import LinkNet
from model.SegNet import SegNet
from model.UNet import UNet
from model.ENet import ENet
from model.ERFNet import ERFNet
from model.CGNet import CGNet
from model.EDANet import EDANet
from model.ESNet import ESNet
from model.ESPNet import ESPNet
from model.LEDNet import LEDNet
from model.ESPNet_v2.SegmentationModel import EESPNet_Seg
from model.ContextNet import ContextNet
from model.FastSCNN import FastSCNN
from model.DABNet import DABNet
from model.FSSNet import FSSNet
from model.FPENet import FPENet
from model.dinov3_seg import dinov3_vits16_mask2form
from model.segdino.dpy import DPT
from model.attunet import AttU_Net, R2AttU_Net, R2U_Net
from model.unetpp import NestedUNet
from model.multiresunet import MultiResUnet
from model._missformer.segformer import SegFormer
from model._missformer.MISSFormer import MISSFormer
from model.MAE_Seg import MAE_Segmentation
from model.moco_v2_seg import MoCoResNetUNet
from model.simsiam_seg import SimSiam_UNet
from model.aim_seg import AIMv2_UNet





def build_model(model_name, num_classes, img_ch=3, image_size=768,
                backbone=None):
    if model_name == 'SQNet':
        return SQNet(classes=num_classes)
    # ap
    elif model_name == 'dinov3_vits16':
        return dinov3_vits16_mask2form(classes=num_classes)
    elif model_name == 'segdino':
        return DPT(nclass=num_classes, backbone=backbone)
    elif model_name == 'unet':
        return UNet(classes=num_classes)    
    elif model_name == 'mae_seg':
        return MAE_Segmentation(num_classes=num_classes, img_size=image_size)  
    elif model_name == 'mocov2_seg':
        return MoCoResNetUNet(num_classes=num_classes)      
    elif model_name == 'simsiam_seg':
        return SimSiam_UNet(num_classes=num_classes)   
    elif model_name == 'aim_seg':
        return AIMv2_UNet(num_classes=num_classes)  



    elif model_name == 'segnet':
        return SegNet(classes=num_classes)
    elif model_name == 'attunet':
        return AttU_Net(img_ch=img_ch, output_ch=num_classes)    
    elif model_name == 'r2unet':
        return R2U_Net(img_ch=img_ch, output_ch=num_classes)    
    elif model_name == 'r2attunet':
        return R2AttU_Net(img_ch=img_ch, output_ch=num_classes)            
    elif model_name == 'unetpp':
        return NestedUNet(input_channels=img_ch, num_classes=num_classes)      
    elif model_name == 'multiresunet':
        return MultiResUnet(channels=img_ch, nclasses=num_classes)     
    elif model_name == 'segformer':
        return SegFormer(num_classes=num_classes, image_size=image_size)  
    elif model_name == 'missformer':
        return  MISSFormer(num_classes=num_classes, in_ch=img_ch, token_mlp_mode="mix_skip", encoder_pretrained=True)

    elif model_name == 'LinkNet':
        return LinkNet(classes=num_classes)


    elif model_name == 'ENet':
        return ENet(classes=num_classes)
    elif model_name == 'ERFNet':
        return ERFNet(classes=num_classes)
    elif model_name == 'CGNet':
        return CGNet(classes=num_classes)
    elif model_name == 'EDANet':
        return EDANet(classes=num_classes)
    elif model_name == 'ESNet':
        return ESNet(classes=num_classes)
    elif model_name == 'ESPNet':
        return ESPNet(classes=num_classes)
    elif model_name == 'LEDNet':
        return LEDNet(classes=num_classes)
    elif model_name == 'ESPNet_v2':
        return EESPNet_Seg(classes=num_classes)
    elif model_name == 'ContextNet':
        return ContextNet(classes=num_classes)
    elif model_name == 'FastSCNN':
        return FastSCNN(classes=num_classes)
    elif model_name == 'DABNet':
        return DABNet(classes=num_classes)
    elif model_name == 'FSSNet':
        return FSSNet(classes=num_classes)
    elif model_name == 'FPENet':
        return FPENet(classes=num_classes)