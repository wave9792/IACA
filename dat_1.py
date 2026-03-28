import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple

from dat_blocks import *


class TransformerStage(nn.Module):

    def __init__(self, fmap_size, window_size, ns_per_pt,
                 dim_in, dim_embed, depths, stage_spec, n_groups,
                 use_pe, sr_ratio,
                 heads, stride, offset_range_factor,
                 dwc_pe, no_off, fixed_pe,
                 attn_drop, proj_drop, expansion, drop, use_dwc_mlp):

        super().__init__()
        fmap_size = to_2tuple(fmap_size)
        self.depths = depths
        hc = dim_embed // heads
        assert dim_embed == heads * hc
        self.proj = nn.Conv2d(dim_in, dim_embed, 1, 1, 0) if dim_in != dim_embed else nn.Identity()

        self.layer_norms = nn.ModuleList(
            [LayerNormProxy(dim_embed) for _ in range(2 * depths)]
        )
        self.mlps = nn.ModuleList(
            [
                TransformerMLPWithConv(dim_embed, expansion, drop)  # all use this
                if use_dwc_mlp else TransformerMLP(dim_embed, expansion, drop)
                for _ in range(depths)
            ]
        )
        self.attns = nn.ModuleList()
        self.drop_path = nn.ModuleList()
        # stage_spec=[['L', 'D'], ['L', 'D'], ['L', 'D', 'L', 'D', 'L', 'D'], ['L', 'D']],
        for i in range(depths):
            if stage_spec[i] == 'L':
                self.attns.append(
                    LocalAttention(dim_embed, heads, window_size, attn_drop, proj_drop)
                )
            elif stage_spec[i] == 'D':
                self.attns.append(
                    DAttentionBaseline(fmap_size, fmap_size, heads,
                                       hc, n_groups, attn_drop, proj_drop,
                                       stride, offset_range_factor, use_pe, dwc_pe,
                                       no_off, fixed_pe)
                )
            elif stage_spec[i] == 'S':
                shift_size = math.ceil(window_size / 2)
                self.attns.append(
                    ShiftWindowAttention(dim_embed, heads, window_size, attn_drop, proj_drop, shift_size, fmap_size)
                )
            else:
                raise NotImplementedError(f'Spec: {stage_spec[i]} is not supported.')
            drop_path_rate = 0.0
            # self.drop_path.append(DropPath(drop_path_rate[i]) if drop_path_rate[i] > 0.0 else nn.Identity())
            self.drop_path.append(DropPath(drop_path_rate))

    def forward(self, x):

        x = self.proj(x)

        positions = []
        references = []
        for d in range(self.depths):
            x0 = x
            #x, pos, ref = self.attns[d](x)

            x, pos, ref = self.attns[d](self.layer_norms[2 * d](x))
            #x, pos, ref = self.attns[d](x)
            # x = self.drop_path[d](x) + x0.

            # x0 = x
            # x = self.mlps[d](self.layer_norms[2 * d + 1](x))
            # x = self.drop_path[d](x) + x0
            # positions.append(pos)
            # references.append(ref)
        return x, positions, references


class DAT(nn.Module):
    def __init__(self, expansion=4,
                 dims=1024, depths=1,
                 heads=16,
                 window_sizes=7,
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
                 strides=1, offset_range_factor=4,
                 stage_spec=['D'],
                 groups=4,
                 use_pes=True,
                 dwc_pes=False,
                 sr_ratios=-1,
                 fixed_pes=False,
                 no_offs=False,
                 ns_per_pts=4,
                 use_dwc_mlps=False,
                 use_conv_patches=False,
                 **kwargs):
        super().__init__()

        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(22))]# the number is 0, total 24
        image_size = 7
        dim1 = dims
        dim2 = dims
        self.stages = TransformerStage(image_size, window_sizes, ns_per_pts,
                                       dim1, dim2, depths, stage_spec, groups, use_pes,
                                       sr_ratios, heads, strides,
                                       offset_range_factor,
                                       dwc_pes, no_offs, fixed_pes,
                                       attn_drop_rate, drop_rate, expansion, drop_rate,
                                       use_dwc_mlps)

        # self.down_projs = nn.Sequential(
        #     # nn.Conv2d(dims, dims, 3, 2, 1, bias=False),
        #     LayerNormProxy(dims)
        # )
        self.reset_parameters()

    def reset_parameters(self):

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # @torch.no_grad()
    # def load_pretrained(self, state_dict):
    #
    #     new_state_dict = {}
    #     for state_key, state_value in state_dict.items():
    #         keys = state_key.split('.')
    #         m = self
    #         for key in keys:
    #             if key.isdigit():
    #                 m = m[int(key)]
    #             else:
    #                 m = getattr(m, key)
    #         if m.shape == state_value.shape:
    #             new_state_dict[state_key] = state_value
    #         else:
    #             # Ignore different shapes
    #             if 'relative_position_index' in keys:
    #                 new_state_dict[state_key] = m.data
    #             if 'q_grid' in keys:
    #                 new_state_dict[state_key] = m.data
    #             if 'reference' in keys:
    #                 new_state_dict[state_key] = m.data
    #             # Bicubic Interpolation
    #             if 'relative_position_bias_table' in keys:
    #                 n, c = state_value.size()
    #                 l = int(math.sqrt(n))
    #                 assert n == l ** 2
    #                 L = int(math.sqrt(m.shape[0]))
    #                 pre_interp = state_value.reshape(1, l, l, c).permute(0, 3, 1, 2)
    #                 post_interp = F.interpolate(pre_interp, (L, L), mode='bicubic')
    #                 new_state_dict[state_key] = post_interp.reshape(c, L ** 2).permute(1, 0)
    #             if 'rpe_table' in keys:
    #                 c, h, w = state_value.size()
    #                 C, H, W = m.data.size()
    #                 pre_interp = state_value.unsqueeze(0)
    #                 post_interp = F.interpolate(pre_interp, (H, W), mode='bicubic')
    #                 new_state_dict[state_key] = post_interp.squeeze(0)
    #
    #     self.load_state_dict(new_state_dict, strict=False)
    #
    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {'absolute_pos_embed'}
    #
    # @torch.jit.ignore
    # def no_weight_decay_keywords(self):
    #     return {'relative_position_bias_table', 'rpe_table'}

    def forward(self, x):
        # x = self.patch_proj(x)#1 128 56 56
        # print('patch_proj',x.shape)
        # positions = []
        # references = []
        # x = self.down_projs(x)  # only layerNorm
        x, pos, ref = self.stages(x)
        # x = self.down_projs(x)#only layerNorm
        # positions.append(pos)
        # references.append(ref)
        return x


if __name__ == '__main__':
    x = torch.randn(1, 1024, 7, 7)
    model = DAT()
    x = model(x)
    print(x.shape)

