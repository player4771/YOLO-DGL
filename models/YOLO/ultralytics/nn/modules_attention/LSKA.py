import torch
import torch.nn as nn
import math
from timm.layers import DropPath, to_2tuple, trunc_normal_


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(_init_weights)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LSKblock(nn.Module):#原名LSKA
    def __init__(self, dim, k_size=7):
        super().__init__()
        self.k_size = k_size

        if k_size == 7:
            k0h, k0v = (1, 3), (3, 1)
            p0h, p0v = (0, 1), (1, 0)
            ksh, ksv = (1, 3), (3, 1)
            psh, psv = (0, 2), (2, 0)
            d = 2
        elif k_size == 11:
            k0h, k0v = (1, 3), (3, 1)
            p0h, p0v = (0, 1), (1, 0)
            ksh, ksv = (1, 5), (5, 1)
            psh, psv = (0, 4), (4, 0)
            d = 2
        elif k_size == 23:
            k0h, k0v = (1, 5), (5, 1)
            p0h, p0v = (0, 2), (2, 0)
            ksh, ksv = (1, 7), (7, 1)
            psh, psv = (0, 9), (9, 0)
            d = 3
        elif k_size == 35:
            k0h, k0v = (1, 5), (5, 1)
            p0h, p0v = (0, 2), (2, 0)
            ksh, ksv = (1, 11), (11, 1)
            psh, psv = (0, 15), (15, 0)
            d = 3
        elif k_size == 41:
            k0h, k0v = (1, 5), (5, 1)
            p0h, p0v = (0, 2), (2, 0)
            ksh, ksv = (1, 13), (13, 1)
            psh, psv = (0, 18), (18, 0)
            d = 3
        elif k_size == 53:
            k0h, k0v = (1, 5), (5, 1)
            p0h, p0v = (0, 2), (2, 0)
            ksh, ksv = (1, 17), (17, 1)
            psh, psv = (0, 24), (24, 0)
            d = 3
        else:
            raise NotImplementedError(f"Invalid k_size {k_size}")

        self.conv0h = nn.Conv2d(dim, dim, kernel_size=k0h, stride=(1, 1), padding=p0h, groups=dim)
        self.conv0v = nn.Conv2d(dim, dim, kernel_size=k0v, stride=(1, 1), padding=p0v, groups=dim)
        self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=ksh, stride=(1, 1), padding=psh, groups=dim, dilation=d)
        self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=ksv, stride=(1, 1), padding=psv, groups=dim, dilation=d)

        self.conv1 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()
        attn = self.conv0h(x)
        attn = self.conv0v(attn)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)
        attn = self.conv1(attn)
        return u * attn



class Attention(nn.Module):
    def __init__(self, d_model, k_size):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LSKblock(d_model, k_size)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        return x


class Block(nn.Module):
    def __init__(self, dim, k_size, mlp_ratio=4., drop=0.,drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim, k_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones(dim), requires_grad=True)

        self.apply(_init_weights)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding"""

    def __init__(self, in_chans=3, embed_dim=768, patch_size=7, stride=4):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.BatchNorm2d(embed_dim)

        self.apply(_init_weights)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)        
        return x, H, W


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x
