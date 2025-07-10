# Copyright (c) OpenMMLab. All rights reserved.
import e2cnn.nn as enn
from e2cnn import gspaces
import torch
from e2cnn.nn import GeometricTensor


N = 8
gspace = gspaces.Rot2dOnR2(N=N)


def build_enn_divide_feature(planes):
    """build an enn regular feature map with the specified number of channels
    divided by N."""
    assert gspace.fibergroup.order() > 0
    N = gspace.fibergroup.order()
    planes = planes / N
    planes = int(planes)
    return enn.FieldType(gspace, [gspace.regular_repr] * planes)


def build_enn_feature(planes):
    """build a enn regular feature map with the specified number of
    channels."""
    return enn.FieldType(gspace, planes * [gspace.regular_repr])


def build_enn_trivial_feature(planes):
    """build a enn trivial feature map with the specified number of
    channels."""
    return enn.FieldType(gspace, planes * [gspace.trivial_repr])


def build_enn_norm_layer(num_features, postfix=''):
    """build an enn normalizion layer."""
    in_type = build_enn_divide_feature(num_features)
    return 'bn' + str(postfix), enn.InnerBatchNorm(in_type)


def ennConv(inplanes,
            outplanes,
            kernel_size=3,
            stride=1,
            padding=0,
            groups=1,
            bias=False,
            dilation=1):
    """enn convolution.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        kernel_size (int, optional): The size of kernel.
        stride (int, optional): Stride of the convolution. Default: 1.
        padding (int or tuple): Zero-padding added to both sides of the input.
            Default: 0.
        groups (int): Number of blocked connections from input.
            channels to output channels. Default: 1.
        bias (bool): If True, adds a learnable bias to the output.
            Default: False.
        dilation (int or tuple): Spacing between kernel elements. Default: 1.
    """
    in_type = build_enn_divide_feature(inplanes)
    out_type = build_enn_divide_feature(outplanes)
    return enn.R2Conv(
        in_type,
        out_type,
        kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        dilation=dilation,
        sigma=None,
        frequencies_cutoff=lambda r: 3 * r,
        # initialize=False,
    )


def ennTrivialConv(inplanes,
                   outplanes,
                   kernel_size=3,
                   stride=1,
                   padding=0,
                   groups=1,
                   bias=False,
                   dilation=1):
    """enn convolution with trivial input featurn.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        kernel_size (int, optional): The size of kernel.
        stride (int, optional): Stride of the convolution. Default: 1.
        padding (int or tuple): Zero-padding added to both sides of the input.
            Default: 0.
        groups (int): Number of blocked connections from input.
            channels to output channels. Default: 1.
        bias (bool): If True, adds a learnable bias to the output.
            Default: False.
        dilation (int or tuple): Spacing between kernel elements. Default: 1.
    """

    in_type = build_enn_trivial_feature(inplanes)
    out_type = build_enn_divide_feature(outplanes)
    return enn.R2Conv(
        in_type,
        out_type,
        kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        dilation=dilation,
        sigma=None,
        frequencies_cutoff=lambda r: 3 * r,
    )


def ennReLU(inplanes):
    """enn ReLU."""
    in_type = build_enn_divide_feature(inplanes)
    return enn.ReLU(in_type, inplace=True)


def ennSiLU(inplanes):
    """enn SiLU."""
    from mmyolo.models.layers.enn_bricks import SiLU
    in_type = build_enn_divide_feature(inplanes)
    return SiLU(in_type, inplace=True)


def ennAvgPool(inplanes,
               kernel_size=1,
               stride=None,
               padding=0,
               ceil_mode=False):
    """enn Average Pooling.

    Args:
        inplanes (int): The number of input channel.
        kernel_size (int, optional): The size of kernel.
        stride (int, optional): Stride of the convolution. Default: 1.
        padding (int or tuple): Zero-padding added to both sides of the input.
            Default: 0.
        ceil_mode (bool, optional): if True, keep information in the corner of
            feature map.
    """
    in_type = build_enn_divide_feature(inplanes)
    return enn.PointwiseAvgPool(
        in_type,
        kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode)


def ennMaxPool(inplanes, kernel_size, stride=1, padding=0):
    """enn Max Pooling."""
    in_type = build_enn_divide_feature(inplanes)
    return enn.PointwiseMaxPool(
        in_type, kernel_size=kernel_size, stride=stride, padding=padding)


def ennInterpolate(inplanes,
                   scale_factor,
                   mode='bilinear',
                   align_corners=True):
    """enn Interpolate."""
    in_type = build_enn_divide_feature(inplanes)
    return enn.R2Upsampling(
        in_type, scale_factor, mode=mode, align_corners=align_corners)


def ennRearrange(enn_tensor: torch.Tensor):
    """enn feature rearrange."""
    channel = enn_tensor.shape[1]
    assert channel % N == 0, f'channel must be divisible by N: {channel}'

    remainders = [(i, i % N) for i in range(channel)]
    grouped_channels = {}
    for idx, rem in remainders:
        if rem not in grouped_channels:
            grouped_channels[rem] = []
        grouped_channels[rem].append(idx)

    new_order = []
    for rem in range(N):
        if rem in grouped_channels:
            new_order.extend(grouped_channels[rem])

    return enn_tensor[:, new_order, :, :]



def ennConcat(enn_tensors: list):
    """enn feature map channel Concat."""
    tensors = []
    for enn_tensor in enn_tensors:
        tensors.append(enn_tensor.tensor)
    result = torch.cat(tensors, dim=1)
    type = build_enn_divide_feature(result.shape[1])
    return GeometricTensor(result, type)


