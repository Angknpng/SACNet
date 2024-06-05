from Code.lib.Swin import SwinTransformer
from torch import nn, Tensor
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from typing import Optional
from torchvision.ops import DeformConv2d
from Code.lib.adaWin import SwinTransformerBlock
import numpy as np
# from modules import DeformConv
import copy

def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

#model
class TMSOD(nn.Module):
    def __init__(self):
        super(TMSOD, self).__init__()
        # self.depth = nn.Conv2d(1, 3, kernel_size=1)
        self.rgb_swin = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
        self.t_swin = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])

        self.MSA_sem = GMSA_ini(d_model=1024)
        self.conv_sem = conv3x3_bn_relu(1024*2, 1024)

        self.MSA4_r = SwinTransformerBlock(dim=1024, input_resolution=(12,12), num_heads=2, up_ratio=1, out_channels=1024)#后续考虑不同层设置不同的window size
        self.MSA4_t = SwinTransformerBlock(dim=1024, input_resolution=(12, 12), num_heads=2, up_ratio=1,
                                         out_channels=1024)
        # self.MSA4_2_r = GMSA_ini(d_model=1024)
        # self.MSA4_2_t = GMSA_ini(d_model=1024)
        self.MSA3_r = SwinTransformerBlock(dim=512, input_resolution=(24,24), num_heads=2, up_ratio=2, out_channels=512)
        self.MSA3_t = SwinTransformerBlock(dim=512, input_resolution=(24, 24), num_heads=2, up_ratio=2, out_channels=512)
        # self.MSA3_2_r = GMSA_ini(d_model=512)
        # self.MSA3_2_t = GMSA_ini(d_model=512)
        self.MSA2_r = SwinTransformerBlock(dim=256, input_resolution=(48,48), num_heads=2, up_ratio=4, out_channels=256)
        self.MSA2_t = SwinTransformerBlock(dim=256, input_resolution=(48, 48), num_heads=2, up_ratio=4, out_channels=256)
        # self.MSA2_2_r = GMSA_ini(d_model=256)
        # self.MSA2_2_t = GMSA_ini(d_model=256)

        self.align_att4 = get_aligned_feat(inC=1024, outC=1024)
        self.align_att3 = get_aligned_feat(inC=512, outC=512)
        self.align_att2 = get_aligned_feat(inC=256, outC=256)

        self.convAtt4 = conv3x3_bn_relu(1024*2, 1024)
        self.convAtt3 = conv3x3_bn_relu(512*2, 512)
        self.convAtt2 = conv3x3_bn_relu(256*2, 256)

        # self.convrt4 = BasicConv2d(1024 * 2, 1024, 1)
        # self.convrt3 = BasicConv2d(512 * 2, 512, 1)
        # self.convrt2 = BasicConv2d(256 * 2, 256, 1)
        # self.convrt1 = BasicConv2d(128 * 2, 128, 1)

        self.conv1024 = conv3x3_bn_relu(1024, 512)
        self.conv512 = conv3x3_bn_relu(512, 256)
        self.conv256 = conv3x3_bn_relu(256, 128)
        self.conv128 = conv3x3_bn_relu(128, 64)
        self.conv64 = conv3x3(64, 1)

        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)

    def forward(self, rgb, t):
        #if depth
        # t = self.depth(t)
        fr = self.rgb_swin(rgb)#[0-3]
        ft = self.t_swin(t)

        semantic = self.MSA_sem(torch.cat((fr[3].flatten(2).transpose(1, 2), ft[3].flatten(2).transpose(1, 2)), dim=1), torch.cat((fr[3].flatten(2).transpose(1, 2), ft[3].flatten(2).transpose(1, 2)), dim=1))#(b,c,hw)->(b,hw,c), cat in hw for att, which contains self and cross.
        semantic1, semantic2 = torch.split(semantic, fr[3].shape[2] * fr[3].shape[3], dim=1)
        semantic = self.conv_sem(torch.cat((semantic1.view(semantic1.shape[0], int(np.sqrt(semantic1.shape[1])), int(np.sqrt(semantic1.shape[1])), -1).permute(0, 3, 1, 2).contiguous(), semantic2.view(semantic2.shape[0], int(np.sqrt(semantic2.shape[1])), int(np.sqrt(semantic2.shape[1])), -1).permute(0, 3, 1, 2).contiguous()), dim=1))

        att_4_r = self.MSA4_r(fr[3].flatten(2).transpose(1, 2), ft[3].flatten(2).transpose(1, 2), semantic=semantic)
        att_4_t = self.MSA4_t(ft[3].flatten(2).transpose(1, 2), fr[3].flatten(2).transpose(1, 2), semantic=semantic)
        att_3_r = self.MSA3_r(fr[2].flatten(2).transpose(1, 2), ft[2].flatten(2).transpose(1, 2), semantic=semantic)
        att_3_t = self.MSA3_t(ft[2].flatten(2).transpose(1, 2), fr[2].flatten(2).transpose(1, 2), semantic=semantic)
        att_2_r = self.MSA2_r(fr[1].flatten(2).transpose(1, 2), ft[1].flatten(2).transpose(1, 2), semantic=semantic)
        att_2_t = self.MSA2_t(ft[1].flatten(2).transpose(1, 2), fr[1].flatten(2).transpose(1, 2), semantic=semantic)
        r1 = fr[0] + ft[0]

        r4 = att_4_r.view(att_4_r.shape[0], int(np.sqrt(att_4_r.shape[1])), int(np.sqrt(att_4_r.shape[1])), -1).permute(0, 3, 1,
                                                                                                                2).contiguous()
        align_t4 = self.align_att4(r4, att_4_t.view(att_4_t.shape[0], int(np.sqrt(att_4_t.shape[1])), int(np.sqrt(att_4_t.shape[1])), -1).permute(0, 3, 1,
                                                                                                                2).contiguous())  # .transpose(1, 2).contiguous().view(fr[3].shape[0], fr[3].shape[1], fr[3].shape[2], fr[3].shape[3])
        #!!!!!!!!!!!!!!!!!后续尝试相乘相加后cat!!!!!!!!!!!!!!!!!!!!
        r4 = self.convAtt4(torch.cat((r4, align_t4), dim=1))
        r3 = att_3_r.view(att_3_r.shape[0], int(np.sqrt(att_3_r.shape[1])), int(np.sqrt(att_3_r.shape[1])), -1).permute(0, 3, 1,
                                                                                                                2).contiguous()
        align_t3 = self.align_att3(r3, att_3_t.view(att_3_t.shape[0], int(np.sqrt(att_3_t.shape[1])), int(np.sqrt(att_3_t.shape[1])), -1).permute(0, 3, 1,
                                                                                                                2).contiguous())  # .transpose(1, 2).contiguous().view(fr[2].shape[0], fr[2].shape[1], fr[2].shape[2], fr[2].shape[3])
        r3 = self.convAtt3(torch.cat((r3, align_t3), dim=1))
        r2 = att_2_r.view(att_2_r.shape[0], int(np.sqrt(att_2_r.shape[1])), int(np.sqrt(att_2_r.shape[1])), -1).permute(0, 3, 1,
                                                                                                                2).contiguous()
        align_t2 = self.align_att2(r2, att_2_t.view(att_2_t.shape[0], int(np.sqrt(att_2_t.shape[1])), int(np.sqrt(att_2_t.shape[1])), -1).permute(0, 3, 1,
                                                                                                                2).contiguous())  # .transpose(1, 2).contiguous().view(fr[1].shape[0], fr[1].shape[1], fr[1].shape[2], fr[1].shape[3])
        r2 = self.convAtt2(torch.cat((r2, align_t2), dim=1))
        # r1 = att_1.transpose(1, 2).contiguous().view(fr[0].shape[0], fr[0].shape[1], fr[0].shape[2], fr[0].shape[3])

        r4 = self.conv1024(self.up2(r4))
        r3 = self.conv512(self.up2(r3 + r4))
        r2 = self.conv256(self.up2(r2 + r3))
        r1 = self.conv128(r1 + r2)
        out = self.up4(r1)
        out = self.conv64(out)
        return out

    def load_pre(self, pre_model):
        self.rgb_swin.load_state_dict(torch.load(pre_model)['model'],strict=False)
        print(f"RGB SwinTransformer loading pre_model ${pre_model}")
        self.t_swin.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"Depth SwinTransformer loading pre_model ${pre_model}")

class GMSA_ini(nn.Module):
    def __init__(self, d_model=256, num_layers=2, decoder_layer=None):
        super(GMSA_ini, self).__init__()
        if decoder_layer is None:
            decoder_layer = GMSA_layer_ini(d_model=d_model, nhead=8)
        self.layers = _get_clones(decoder_layer, num_layers)
    def forward(self, fr, ft):
        # fr = fr.flatten(2).transpose(1, 2)  # b hw c
        # ft = ft.flatten(2).transpose(1, 2)
        output = fr
        for layer in self.layers:
            output = layer(output, ft)
        return output
class GMSA_layer_ini(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(GMSA_layer_ini, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.sigmoid = nn.Sigmoid()
    def forward(self, fr, ft, pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):


        fr2 = self.multihead_attn(query=self.with_pos_embed(fr, query_pos).transpose(0, 1),#hw b c
                                   key=self.with_pos_embed(ft, pos).transpose(0, 1),
                                   value=ft.transpose(0, 1))[0].transpose(0, 1)#b hw c
        fr = fr + self.dropout2(fr2)
        fr = self.norm2(fr)

        fr2 = self.linear2(self.dropout(self.activation(self.linear1(fr))))  #FFN
        fr = fr + self.dropout3(fr2)
        fr = self.norm3(fr)
        # print(fr.shape)
        return fr

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

#gated MSA
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# gated MSA layer
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class get_aligned_feat(nn.Module):
    def __init__(self, inC, outC):
        super(get_aligned_feat, self).__init__()
        self.deformConv1 = defomableConv(inC=inC*2, outC=outC)
        self.deformConv2 = defomableConv(inC=inC, outC=outC)
        self.deformConv3 = defomableConv(inC=inC, outC=outC)
        self.deformConv4 = defomableConv_offset(inC=inC, outC=outC)

    def forward(self, fr, ft):
        cat_feat = torch.cat((fr, ft), dim=1)
        feat1 = self.deformConv1(cat_feat)
        feat2 = self.deformConv2(feat1)
        feat3 = self.deformConv3(feat2)
        aligned_feat = self.deformConv4(feat3, ft)
        return aligned_feat

class defomableConv(nn.Module):
    def __init__(self, inC, outC, kH = 3, kW = 3, num_deformable_groups = 4):
        super(defomableConv, self).__init__()
        self.num_deformable_groups = num_deformable_groups
        self.inC = inC
        self.outC = outC
        self.kH = kH
        self.kW = kW
        self.offset = nn.Conv2d(inC, num_deformable_groups * 2 * kH * kW, kernel_size=(kH, kW), stride=(1, 1), padding=(1, 1), bias=False)
        self.deform = DeformConv2d(inC, outC, (kH, kW), stride=1, padding=1, groups=num_deformable_groups)

    def forward(self, x):
        offset = self.offset(x)
        out = self.deform(x, offset)
        return out

class defomableConv_offset(nn.Module):
    def __init__(self, inC, outC, kH = 3, kW = 3, num_deformable_groups = 2):
        super(defomableConv_offset, self).__init__()
        self.num_deformable_groups = num_deformable_groups
        self.inC = inC
        self.outC = outC
        self.kH = kH
        self.kW = kW
        self.offset = nn.Conv2d(inC, num_deformable_groups * 2 * kH * kW, kernel_size=(kH, kW), stride=(1, 1), padding=(1, 1), bias=False)
        self.deform = DeformConv2d(inC, outC, (kH, kW), stride=1, padding=1, groups=num_deformable_groups)

    def forward(self, feta3, x):
        offset = self.offset(feta3)
        out = self.deform(x, offset)
        return out