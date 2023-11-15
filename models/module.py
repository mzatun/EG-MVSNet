import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import gaussian
import numpy as np

class ConvGRUCell2(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size):
        super(ConvGRUCell2, self).__init__()

        # filters used for gates
        gru_input_channel = input_channel + output_channel
        self.output_channel = output_channel

        self.gate_conv = nn.Conv2d(gru_input_channel, output_channel * 2, kernel_size, padding=1)
        self.reset_gate_norm = nn.GroupNorm(1, output_channel, 1e-5, True)
        self.update_gate_norm = nn.GroupNorm(1, output_channel, 1e-5, True)

        # filters used for outputs
        self.output_conv = nn.Conv2d(gru_input_channel, output_channel, kernel_size, padding=1)
        self.output_norm = nn.GroupNorm(1, output_channel, 1e-5, True)

        self.activation = nn.Tanh()

    def gates(self, x, h):
        # x = N x C x H x W
        # h = N x C x H x W

        # c = N x C*2 x H x W
        c = torch.cat((x, h), dim=1)
        f = self.gate_conv(c)

        # r = reset gate, u = update gate
        # both are N x O x H x W
        C = f.shape[1]
        r, u = torch.split(f, C // 2, 1)

        rn = self.reset_gate_norm(r)
        un = self.update_gate_norm(u)
        rns = F.sigmoid(rn)
        uns = F.sigmoid(un)
        return rns, uns

    def output(self, x, h, r, u):
        f = torch.cat((x, r * h), dim=1)
        o = self.output_conv(f)
        on = self.output_norm(o)
        return on

    def forward(self, x, h = None):
        N, C, H, W = x.shape
        HC = self.output_channel
        if(h is None):
            h = torch.zeros((N, HC, H, W), dtype=torch.float, device=x.device)
        r, u = self.gates(x, h)
        o = self.output(x, h, r, u)
        y = self.activation(o)
        output = u * h + (1 - u) * y
        return output, output


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super(ConvGRUCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)

        self.conv_gates = nn.Conv2d(self.input_channels + self.hidden_channels, 2 * self.hidden_channels,
                                    kernel_size=self.kernel_size, stride=1,
                                    padding=self.padding, bias=True)
        self.convc = nn.Conv2d(self.input_channels + self.hidden_channels, self.hidden_channels,
                               kernel_size=self.kernel_size, stride=1,
                               padding=self.padding, bias=True)

    def forward(self, x, h):
        N, C, H, W = x.shape[0],x.shape[1],x.shape[2], x.shape[3]
        HC = self.hidden_channels
        if(h is None):
            h = torch.zeros((N, HC, H, W), dtype=torch.float, device=x.device)

        input = torch.cat((x, h), dim=1)
        gates = self.conv_gates(input)

        reset_gate, update_gate = torch.chunk(gates, dim=1, chunks=2)

        # activation
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)

        # print(reset_gate)
        # concatenation
        input = torch.cat((x, reset_gate * h), dim=1)

        # convolution
        conv = self.convc(input)

        # activation
        conv = torch.tanh(conv)

        # soft update
        output = update_gate * h + (1 - update_gate) * conv

        return output, output


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)

    def forward(self, x):
        return F.relu(self.conv(x), inplace=True)

class Deconv2d(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))

class ConvTransBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, output_pad=1):
        super(ConvTransBnReLU, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad, output_padding=output_pad, bias=False)

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

class ConvTransReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, output_pad=1):
        super(ConvTransReLU, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad, output_padding=output_pad, bias=False)


    def forward(self, x):
        return F.relu(self.conv(x), inplace=True)


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvGnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvGnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        G = max(1, output_channel // 8)
        self.gn = nn.GroupNorm(G, out_channels)

    def forward(self, x):
        return F.relu(self.gn(self.conv(x)), inplace=True)


class ConvGn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvGn, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        G = max(1, output_channel // 8)
        self.gn = nn.GroupNorm(G, out_channels)

    def forward(self, x):
        return self.gn(self.conv(x))

class ConvTransGnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, output_pad=1):
        super(ConvTransGnReLU, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad, output_padding=output_pad, bias=False)
        G = max(1, output_channel // 8)
        self.gn = nn.GroupNorm(G, out_channels)

    def forward(self, x):
        return F.relu(self.gn(self.conv(x)), inplace=True)



class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=stride, pad=1)
        self.conv2 = ConvBn(out_channels, out_channels, kernel_size=3, stride=1, pad=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


class Hourglass3d(nn.Module):
    def __init__(self, channels):
        super(Hourglass3d, self).__init__()

        self.conv1a = ConvBnReLU3D(channels, channels * 2, kernel_size=3, stride=2, pad=1)
        self.conv1b = ConvBnReLU3D(channels * 2, channels * 2, kernel_size=3, stride=1, pad=1)

        self.conv2a = ConvBnReLU3D(channels * 2, channels * 4, kernel_size=3, stride=2, pad=1)
        self.conv2b = ConvBnReLU3D(channels * 4, channels * 4, kernel_size=3, stride=1, pad=1)

        self.dconv2 = nn.Sequential(
            nn.ConvTranspose3d(channels * 4, channels * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels * 2))

        self.dconv1 = nn.Sequential(
            nn.ConvTranspose3d(channels * 2, channels, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels))

        self.redir1 = ConvBn3D(channels, channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = ConvBn3D(channels * 2, channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1b(self.conv1a(x))
        conv2 = self.conv2b(self.conv2a(conv1))
        dconv2 = F.relu(self.dconv2(conv2) + self.redir2(conv1), inplace=True)
        dconv1 = F.relu(self.dconv1(dconv2) + self.redir1(x), inplace=True)
        return dconv1

class GlobalPoolingModule(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(GlobalPoolingModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv3 = nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1, padding=0,  bias=False)

    def forward(self, x):
        block1 = self.conv1(x)
        block2 = self.conv2(self.avg_pool(block1))
        block3 = self.sigmoid(block2)
        block4 = block1 + block3 * block1
        block5 = self.conv3(block4)

        return block5


class ChannelAttentionModule(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv3 = nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self,x1, x2):
        x = x1 + x2
        block1 = self.avg_pool(x)
        block2 = self.relu(self.conv1(block1))
        block3 = self.sigmoid(self.conv2(block2))
        block4 = x + block3 * x
        block4 = self.conv3(block4)

        return block4


def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    # depth_values = -depth_values
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj)) # Tcw
        #proj = torch.matmul(torch.inverse(src_proj), ref_proj)   # Twc
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                            1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea

def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return

def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return

class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class Conv2dUnit(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, **kwargs):
        super(Conv2dUnit, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

class AtrousConv2dUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1,
                 relu=True, bn=True, bn_momentum=0.1, **kwargs):
        super(AtrousConv2dUnit, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(ChannelAttention, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(inchannels)
        self.conv1 = nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = x1 + x2
        # block1 = self.avg_pool(x) # [B, 16, 16, 16]
        block2 = F.relu(self.conv1(x),  inplace=True)
        block3 = self.sigmoid(self.conv2(block2))
        block4 = x + block3 * x

        return block4

class DADownSampleModule(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(DADownSampleModule, self).__init__()
        base_channel = 8
        self.conv0 = Conv2dUnit(inchannels, base_channel, 3, 1, padding=1)
        self.conv0_1 = Conv2dUnit(base_channel, outchannels, 5, 2, padding=2)
        self.conv0_2 = ChannelAttention(outchannels, outchannels)

        self.conv1 = Conv2dUnit(inchannels, base_channel, 3, 1, padding=1)
        self.conv1_1 = nn.Sequential(
            AtrousConv2dUnit(base_channel, base_channel, 3, 1, dilation=2),
            Conv2dUnit(base_channel, outchannels, 5, 2, padding=2),
        )
        self.conv1_2 = ChannelAttention(outchannels, outchannels)

        self.conv2 = Conv2dUnit(inchannels, base_channel, 3, 1, padding=1)
        self.conv2_1 = nn.Sequential(
            AtrousConv2dUnit(base_channel, base_channel, 3, 1, dilation=3),
            Conv2dUnit(base_channel, outchannels, 5, 2, padding=2),
        )
        self.conv2_2 = ChannelAttention(outchannels, outchannels)

        self.final = nn.Sequential(
            Conv2dUnit(outchannels * 3, outchannels, 3, 1, padding=1),
            nn.Conv2d(outchannels, outchannels, 1, 1)
        )

    def forward(self, x): # [B, 8, H, W]
        # branch 0
        x0 = self.conv0(x) # [B, 8, H, W]
        x0_1 = self.conv0_1(x0) # [B, 8, H/2, W/2]
        wx0 = self.conv0_2(x0_1)
        # wx0 = x0_1 + weight0

        # branch 1
        x1 = self.conv1(x)
        x1_1 = self.conv1_1(x1)
        wx1 = self.conv1_2(x1_1)
        # wx1 = x1_1 + weight1

        # branch 2
        x2 = self.conv2(x)
        x2_1 = self.conv2_1(x2)
        wx2 = self.conv2_2(x2_1)
        # wx2 = x2_1 + weight2

        wx = torch.cat([wx0, wx1, wx2], dim=1) # [B, 24, H/2, W/2]
        res = self.final(wx) # [B, 8, H/2, W/2]

        return res

# fixme: Surface Feature Extraction Module
class SFEModule(nn.Module):
    def __init__(self, base_f):
        super(SFEModule, self).__init__()
        base_channels = base_f

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = DADownSampleModule(base_channels, base_channels * 2)

        self.conv1_1 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(16, 16, 3, 1, 1)

    def forward(self, x): # x [B, 3, 384, 768] [B, 3, H, W]
        x = self.conv0(x) # [B, 8, 384, 768] [B, 8, H, W]
        x = self.conv1(x) # [B, 16, 192, 384] [B, 16, H/2, W/2]
        output = self.conv1_2(self.conv1_1(x))

        return output

# fixme: 0° Learnable Sobel Kernel Gradient Convolution
class Learnable_0_SKGConv(nn.Module):
    def __init__(self, input_channel, output_channel, stride=1, kernel_size=3, padding=1, bias=False, ):
        super(Learnable_0_SKGConv, self).__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                          bias=bias)

    def symmetrize(self, conv_wt):
        """ 0° Learnable Sobel Kernel Structure"""
        # conv_wt.data = (conv_wt - conv_wt.flip(2).flip(3)) / 2
        partA = conv_wt - conv_wt.flip(2)
        partB = partA.flip(2).flip(3)
        conv_wt.data = (partA - partB) / 2

    def forward(self, x):
        self.symmetrize(self.conv.weight)
        res = self.conv(x)
        return res

# fixme: 90° Learnable Sobel Kernel Gradient Convolution
class Learnable_90_SKGConv(nn.Module):
    def __init__(self, input_channel, output_channel, stride=1, kernel_size=3, padding=1, bias=False, ):
        super(Learnable_90_SKGConv, self).__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                          bias=bias)

    def symmetrize(self, conv_wt):
        """ 90° Learnable Sobel Kernel Structure"""
        # conv_wt.data = (conv_wt - conv_wt.flip(2).flip(3)) / 2
        partA = conv_wt - conv_wt.flip(3)
        partB = partA.flip(2).flip(3)
        conv_wt.data = (partA - partB) / 2

    def forward(self, x):
        self.symmetrize(self.conv.weight)
        res = self.conv(x)
        return res

class DeConv2dFuse(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, bn=True,
                 bn_momentum=0.1):
        super(DeConv2dFuse, self).__init__()

        self.deconv = Deconv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1,
                               bn=True, relu=relu, bn_momentum=bn_momentum)

        self.conv = Conv2d(2*out_channels, out_channels, kernel_size, stride=1, padding=1,
                           bn=bn, relu=relu, bn_momentum=bn_momentum)

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x_pre, x):
        x = self.deconv(x)
        x = torch.cat((x, x_pre), dim=1)
        x = self.conv(x)
        return x

# fixme: Edge Feature Extraction Module
class EFEModule(nn.Module):
    def __init__(self, base_f):
        super(EFEModule, self).__init__()
        # Canny Part
        filter_size = 5
        generated_filters = gaussian(filter_size, std=1.0).reshape([1, filter_size])

        self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, filter_size),
                                                    padding=(0, filter_size // 2))
        self.gaussian_filter_horizontal.weight.data.copy_(torch.from_numpy(generated_filters))
        self.gaussian_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size, 1),
                                                  padding=(filter_size // 2, 0))
        self.gaussian_filter_vertical.weight.data.copy_(torch.from_numpy(generated_filters.T))
        self.gaussian_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])

        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape,
                                                 padding=sobel_filter.shape[0] // 2)
        self.sobel_filter_horizontal.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.sobel_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape,
                                               padding=sobel_filter.shape[0] // 2)
        self.sobel_filter_vertical.weight.data.copy_(torch.from_numpy(sobel_filter.T))
        self.sobel_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))


        # Conv Part
        base_channels = base_f
        # self.init_conv = nn.Sequential(
        #     Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1),
        #     Conv2d(base_channels, 3, kernel_size=3, stride=1, padding=1)
        # )
        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv1_1 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(16, 16, 3, 1, 1)

    def forward(self, img): # [1, 3, 128,160]
        # Canny Part
        with torch.no_grad():
            img_r = img[:, 0:1]
            img_g = img[:, 1:2]
            img_b = img[:, 2:3]

            blur_horizontal = self.gaussian_filter_horizontal(img_r)
            blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
            blur_horizontal = self.gaussian_filter_horizontal(img_g)
            blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
            blur_horizontal = self.gaussian_filter_horizontal(img_b)
            blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)

            blurred_img = torch.stack([blurred_img_r, blurred_img_g, blurred_img_b], dim=1)
            blurred_img = torch.stack([torch.squeeze(blurred_img)])

            grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
            grad_y_r = self.sobel_filter_vertical(blurred_img_r)
            grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
            grad_y_g = self.sobel_filter_vertical(blurred_img_g)
            grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
            grad_y_b = self.sobel_filter_vertical(blurred_img_b)

            # COMPUTE THICK EDGES

            grad_mag = torch.sqrt(grad_x_r ** 2 + grad_y_r ** 2)
            grad_mag += torch.sqrt(grad_x_g ** 2 + grad_y_g ** 2)
            grad_mag += torch.sqrt(grad_x_b ** 2 + grad_y_b ** 2)

        # Conv Part
        # img: [1, 3, 640, 512]
        # img = torch.cat([img, grad_mag], 1) # [1, 4, 640, 512]
        x = img + grad_mag

        # x = self.init_conv(img) # [1, 3, 640 ,512]
        # edge_feature = self.edge_detector(x)
        # x = torch.cat([edge_feature, x], 1)
        conv0 = self.conv0(x)  # [B, 8, 384, 768] [B, 8, H, W]
        conv1 = self.conv1(conv0)  # [B, 16, 192, 384] [B, 16, H/2, W/2]
        output = self.conv1_2(self.conv1_1(conv1))
        return output

# fixme: Edge Feature Extraction Module(replace 3×3 Conv2d)
class EFEModule_V1(nn.Module):
    def __init__(self, base_f):
        super(EFEModule_V1, self).__init__()
        self.skg_0_conv = Learnable_0_SKGConv(3, 3, stride=1, kernel_size=3, padding=1, bias=True)
        self.skg_90_conv = Learnable_90_SKGConv(3, 3, stride=1, kernel_size=3, padding=1, bias=True)

        # Conv Part
        base_channels = base_f
        # self.init_conv = nn.Sequential(
        #     Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1),
        #     Conv2d(base_channels, 3, kernel_size=3, stride=1, padding=1)
        # )
        self.conv0 = nn.Sequential(
            Conv2d(3 * 2, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv1_1 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(16, 16, 3, 1, 1)


    def forward(self, x):
        x1 = self.skg_0_conv(x)
        x2 = self.skg_90_conv(x)
        x = torch.cat([x1, x2], dim=1)

        conv0 = self.conv0(x)  # [B, 8, 384, 768] [B, 8, H, W]
        conv1 = self.conv1(conv0)  # [B, 16, 192, 384] [B, 16, H/2, W/2]
        output = self.conv1_2(self.conv1_1(conv1))

        return output

class ResnetBlockGn(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bias, group_channel=8):
        super(ResnetBlockGn, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1], padding=((kernel_size-1)//2)*dilation[1], bias=bias),
            nn.GroupNorm(int(max(1, in_channels / group_channel)), in_channels),
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.stem(x) + x
        out = self.relu(out)
        return out

def resnet_block_gn(in_channels,  kernel_size=3, dilation=[1,1], bias=True, group_channel=8):
    return ResnetBlockGn(in_channels, kernel_size, dilation, bias=bias, group_channel=group_channel)

# fixme: Volume Fusion
class VolumeFusionModule(nn.Module):
    def __init__(self, in_channels=32, bias=True):
        super(VolumeFusionModule, self).__init__()
        self.reweight_network1 = nn.Sequential(
            nn.Conv3d(in_channels, 4, kernel_size=1, padding=0),
            resnet_block_gn(4, kernel_size=1),
            nn.Conv3d(4, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.reweight_network2 = nn.Sequential(
            nn.Conv3d(in_channels, 4, kernel_size=1, padding=0),
            resnet_block_gn(4, kernel_size=1),
            nn.Conv3d(4, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        weight1 = self.reweight_network1(x)  # [1, 1, 128, 160]
        weight2 = self.reweight_network2(y)  # [1, 1, 128, 160]
        x = (weight1 + 1) * x  # [1, 32, 128, 160]
        y = (weight2 + 1) * y  # [1, 32, 128, 160]
        res = x + y  # [1, 32, 128, 160]
        return res

class Deconv2dUnit(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, **kwargs):
        super(Deconv2dUnit, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

class Deconv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, bn=True,
                 bn_momentum=0.1):
        super(Deconv2dBlock, self).__init__()

        self.deconv = Deconv2dUnit(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1,
                                   bn=True, relu=relu, bn_momentum=bn_momentum)

        self.conv = Conv2dUnit(2 * out_channels, out_channels, kernel_size, stride=1, padding=1,
                               bn=bn, relu=relu, bn_momentum=bn_momentum)

    def forward(self, x_pre, x):
        x = self.deconv(x)
        x = torch.cat((x, x_pre), dim=1)
        x = self.conv(x)
        return x

# fixme: Edge Regression Net
class EedgeRegNet(nn.Module):
    def __init__(self, in_channels=32, num_stage=3, ):
        super(EedgeRegNet, self).__init__()
        base_channels = 8
        self.conv0 = nn.Sequential(
            Conv2dUnit(in_channels+1, base_channels, 3, 1, padding=1),
            Conv2dUnit(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2dUnit(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2dUnit(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2dUnit(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2dUnit(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2dUnit(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2dUnit(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )

        self.deconv1 = Deconv2dBlock(base_channels * 4, base_channels * 2, 3)
        self.deconv2 = Deconv2dBlock(base_channels * 2, base_channels, 3)

        self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
        self.out3 = nn.Conv2d(base_channels, 1, 1, bias=False)

    def forward(self, ef_re_feature, coarse_depth):
        batch, img_height, img_width = coarse_depth.shape[0], coarse_depth.shape[1], coarse_depth.shape[2]
        ef_re_feature = F.interpolate(ef_re_feature, [img_height, img_width], mode='bilinear')
        x = torch.cat([ef_re_feature, coarse_depth.unsqueeze(1)], dim=1)

        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        intra_feat = conv2
        intra_feat = self.deconv1(conv1, intra_feat)
        out = self.out2(intra_feat)

        intra_feat = self.deconv2(conv0, intra_feat)
        out = self.out3(intra_feat)

        return out

# p: probability volume [B, D, H, W]
# depth_values: discrete depth values [B, D]
def depth_regression(p, depth_values):
    depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)
    return depth


if __name__ == "__main__":
    # some testing code, just IGNORE it
    from datasets import find_dataset_def
    from torch.utils.data import DataLoader
    import numpy as np
    import cv2

    MVSDataset = find_dataset_def("dtu_yao")
    dataset = MVSDataset("/home/xyguo/dataset/dtu_mvs/processed/mvs_training/dtu/", '../lists/dtu/train.txt', 'train',
                         3, 256)
    dataloader = DataLoader(dataset, batch_size=2)
    item = next(iter(dataloader))

    imgs = item["imgs"][:, :, :, ::4, ::4].cuda()
    proj_matrices = item["proj_matrices"].cuda()
    mask = item["mask"].cuda()
    depth = item["depth"].cuda()
    depth_values = item["depth_values"].cuda()

    imgs = torch.unbind(imgs, 1)
    proj_matrices = torch.unbind(proj_matrices, 1)
    ref_img, src_imgs = imgs[0], imgs[1:]
    ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

    warped_imgs = homo_warping(src_imgs[0], src_projs[0], ref_proj, depth_values)

    cv2.imwrite('../tmp/ref.png', ref_img.permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1] * 255)
    cv2.imwrite('../tmp/src.png', src_imgs[0].permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1] * 255)

    for i in range(warped_imgs.shape[2]):
        warped_img = warped_imgs[:, :, i, :, :].permute([0, 2, 3, 1]).contiguous()
        img_np = warped_img[0].detach().cpu().numpy()
        cv2.imwrite('../tmp/tmp{}.png'.format(i), img_np[:, :, ::-1] * 255)


    # generate gt
    def tocpu(x):
        return x.detach().cpu().numpy().copy()


    ref_img = tocpu(ref_img)[0].transpose([1, 2, 0])
    src_imgs = [tocpu(x)[0].transpose([1, 2, 0]) for x in src_imgs]
    ref_proj_mat = tocpu(ref_proj)[0]
    src_proj_mats = [tocpu(x)[0] for x in src_projs]
    mask = tocpu(mask)[0]
    depth = tocpu(depth)[0]
    depth_values = tocpu(depth_values)[0]

    for i, D in enumerate(depth_values):
        height = ref_img.shape[0]
        width = ref_img.shape[1]
        xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
        print("yy", yy.max(), yy.min())
        yy = yy.reshape([-1])
        xx = xx.reshape([-1])
        X = np.vstack((xx, yy, np.ones_like(xx)))
        # D = depth.reshape([-1])
        # print("X", "D", X.shape, D.shape)

        X = np.vstack((X * D, np.ones_like(xx)))
        X = np.matmul(np.linalg.inv(ref_proj_mat), X)
        X = np.matmul(src_proj_mats[0], X)
        X /= X[2]
        X = X[:2]

        yy = X[0].reshape([height, width]).astype(np.float32)
        xx = X[1].reshape([height, width]).astype(np.float32)

        warped = cv2.remap(src_imgs[0], yy, xx, interpolation=cv2.INTER_LINEAR)
        # warped[mask[:, :] < 0.5] = 0

        cv2.imwrite('../tmp/tmp{}_gt.png'.format(i), warped[:, :, ::-1] * 255)
