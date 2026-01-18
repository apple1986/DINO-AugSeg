from monai.networks.nets import SwinUNETR, SegResNet, VNet, UNet, BasicUNetPlusPlus, DAF3D
from monai.networks.nets import DynUNet
from .dynunet_pipeline.create_network import get_kernels_strides
from .dynunet_pipeline.task_params import deep_supr_num


def choice_net(model_name, in_ch, out_ch,):
    if model_name == "SwinUNETR":
        net = SwinUNETR(
            img_size = [256, 256],
            in_channels=in_ch,
            out_channels=out_ch,
            feature_size=48,
            use_checkpoint=True,
            spatial_dims=2,
            )       
    elif model_name == "SegResNet":
        net = SegResNet(
            spatial_dims=2,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=in_ch,
            out_channels=out_ch,
            # dropout_prob=0.2,
        ) 
    elif model_name == "VNet":
        net = VNet(
            in_channels=in_ch,
            out_channels=out_ch,
        )
    
    elif model_name == "UNet":
        net=UNet(
            spatial_dims=2,
            in_channels=in_ch,
            out_channels=out_ch,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
        )
    elif model_name == "BasicUNetPlusPlus":
        net=BasicUNetPlusPlus(
            spatial_dims=3,
            in_channels=in_ch,
            out_channels=out_ch,
        ) 

    elif model_name == "DAF3D":
        net=DAF3D(
            in_channels=in_ch,
            out_channels=out_ch,
        ) 
    elif model_name == "nnUNet":
        task_id = 'ap_busi'
        kernels, strides = get_kernels_strides(task_id)
        net = DynUNet(
            spatial_dims=2,
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:],
            norm_name="instance",
            deep_supervision=False,
            deep_supr_num=deep_supr_num[task_id],
            )

    return net