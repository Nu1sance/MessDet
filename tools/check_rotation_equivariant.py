import torch

import e2cnn.nn as enn
from e2cnn import gspaces

from mmyolo.models.backbones.re_cspnext import RECSPNeXt
from mmyolo.models.necks.re_cspnext_pafpn import RECSPNeXtPAFPN
from mmyolo.models.utils import build_enn_trivial_feature
from mmyolo.models.utils import build_enn_divide_feature


r2_act = gspaces.Rot2dOnR2(N=8)
is_strict = True


def equivariance_error(x, y):
    assert x.shape == y.shape
    assert x.shape[0] == 1
    sum_squared = torch.sum((x - y) ** 2)
    total_elements = x.shape[1] * x.shape[2] * x.shape[3]
    return torch.sqrt(sum_squared) / total_elements



if __name__ == "__main__":
    m1 = RECSPNeXt(is_strict=is_strict, act_cfg=dict(type='SiLU', inplace=True), is_test=True)
    m1.eval()
    m2 = RECSPNeXtPAFPN(in_channels=[256, 512, 1024], out_channels=256, is_strict=is_strict, act_cfg=dict(type='SiLU', inplace=True))
    m2.eval()

    x = torch.randn(1, 3, 512, 512)
    x = enn.GeometricTensor(x, build_enn_trivial_feature(3))
    y = m2(m1(x))[-1]
    y = enn.GeometricTensor(y, build_enn_divide_feature(y.size()[1]))

    for g in r2_act.testing_elements:
        x_transformed = x.transform(g)
        y_from_x_transformed = m2(m1(x_transformed))[-1]
        y_from_x_transformed = enn.GeometricTensor(y_from_x_transformed, build_enn_divide_feature(y_from_x_transformed.size()[1]))
        y_transformed_from_x = y.transform(g)
        print(f'For {g*45} angle rotation, model\'s equivariance:','YES' if torch.allclose(y_from_x_transformed.tensor, y_transformed_from_x.tensor, atol=1e-7) else 'NO')
        print(f'Equivariance Errors: {equivariance_error(y_from_x_transformed.tensor, y_transformed_from_x.tensor)}')

