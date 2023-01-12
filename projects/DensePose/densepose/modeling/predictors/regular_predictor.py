import torch
import torch.nn as nn

from detectron2.layers import Conv2d, ConvTranspose2d, interpolate
import fvcore.nn.weight_init as weight_init
from torch.nn import functional as F
from .registry import DENSEPOSE_PREDICTOR_REGISTRY

def initialize_module_params(module):
    for name, param in module.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0)
        elif "weight" in name:
            nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

@DENSEPOSE_PREDICTOR_REGISTRY.register()
class RegularPredictor(nn.Module):
    def __init__(self, cfg, input_channels):

        super(RegularPredictor, self).__init__()
        dim_in = input_channels
        n_segm_chan = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS
        dim_out_patches = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_PATCHES + 1
        kernel_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECONV_KERNEL
        self.ann_index_lowres = ConvTranspose2d(
            dim_in, n_segm_chan, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )
        self.index_uv_lowres = ConvTranspose2d(
            dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )
        self.u_lowres = ConvTranspose2d(
            dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )
        self.v_lowres = ConvTranspose2d(
            dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )
        self.scale_factor = cfg.MODEL.ROI_DENSEPOSE_HEAD.UP_SCALE
        initialize_module_params(self)

    def forward(self, head_outputs):
        ann_index_lowres = self.ann_index_lowres(head_outputs)
        index_uv_lowres = self.index_uv_lowres(head_outputs)
        u_lowres = self.u_lowres(head_outputs)
        v_lowres = self.v_lowres(head_outputs)

        def interp2d(input):
            return interpolate(
                input, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
            )

        ann_index = interp2d(ann_index_lowres)
        index_uv = interp2d(index_uv_lowres)
        u = interp2d(u_lowres)
        v = interp2d(v_lowres)
        return (
            (ann_index, index_uv, u, v),
            (ann_index_lowres, index_uv_lowres, u_lowres, v_lowres),
        )