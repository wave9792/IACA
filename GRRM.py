import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple
NORM_EPS = 1e-5
from dat_blocks import *

class LayerNormProxy(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')

class DAttentionBaseline(nn.Module):

    def __init__(
            self, q_size, kv_size, n_heads, n_head_channels):
            # q_size 7*7 kv_size 7*7 n_heads 32 n_head_channels 32
        super().__init__()
        self.n_head_channels = n_head_channels
        #gen hao D fen zhi 1
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        #self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = kv_size
        self.nc = n_head_channels * n_heads


        # ksizes = [9, 7, 5, 3]
        # kk = ksizes[stage_idx]
        kk=3

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.nc, self.nc, 3, 1, kk // 2),
            LayerNormProxy(self.nc),
            nn.GELU(),
            nn.Conv2d(self.nc, 2, 1, 1, 0, bias=False)
        )

        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.ref_point14 = nn.Conv2d(
            self.nc, 2,
            kernel_size=1, stride=1, padding=0
        )
        self.conv = nn.Conv2d(
            self.nc, 16,
            kernel_size=3, stride=1, padding=1
        )
        self.rpe_table = nn.Parameter(
            torch.zeros(self.n_heads, self.kv_h * 2 - 1, self.kv_w * 2 - 1)
        )
        trunc_normal_(self.rpe_table, std=0.01)

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):


        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device)
        )

        ref = torch.stack((ref_y, ref_x), -1)

        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B , -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self, x):

        B, C, H, W = x.size() #1 1024 7 7
        dtype, device = x.dtype, x.device
        q = self.proj_q(x)
        offset = self.conv_offset(q)  #1 2 7 7

        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk
        # ?????
        offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1)
        offset = offset.tanh().mul(offset_range)
        offset = einops.rearrange(offset, 'b p h w -> b h w p')

        referencek = self.ref_point14(x) #1 2 7 7

        referencek = torch.mean(referencek, dim=0).unsqueeze(dim=0)
        referencek = einops.rearrange(referencek, 'b p h w -> b h w p').tanh()
        reference = referencek.expand(B, -1, -1, -1)
        #print(reference.shape)
        offset_x = torch.chunk(offset, 2, dim=3)[0].detach().cpu().numpy()
        offset_y = torch.chunk(offset, 2, dim=3)[1].detach().cpu().numpy()
        reference_x = torch.chunk(reference, 2, dim=3)[0].detach().cpu().numpy()
        reference_y = torch.chunk(reference, 2, dim=3)[1].detach().cpu().numpy()

        temp_x = offset_x * reference_x
        temp_y = offset_y * reference_y

        temp_x = np.where(temp_x <= 0.00, 0, 1)
        temp_y = np.where(temp_y <= 0.00, 0, 1)
        offset_temp = temp_x * temp_y
        manu_offset = np.where(offset_temp <= 0.00, 0.25, 1)

        manu_offset_ = torch.cat((torch.from_numpy(manu_offset), torch.from_numpy(manu_offset)), 3).float().to(device)

        offset = offset * manu_offset_

        pos = offset + reference # 1 7 7 2
        x_sampled = F.grid_sample(
            input=x.reshape(B, self.nc, H, W),
            grid=pos[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg

        x_sampled = x_sampled.reshape(B, C, 1, n_sample) #1 1024 1 49

        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        # yi xia jiu shi lun wen zhong gong shi (1) de shi xian
        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)

        rpe_table = self.rpe_table
        rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)

        q_grid = self._get_ref_points(H, W, B, dtype, device)

        displacement = (
                q_grid.reshape(B , H * W, 2).unsqueeze(2) - pos.reshape(B ,n_sample,2).unsqueeze(1)).mul(0.5)
        attn_bias = F.grid_sample(
            input=rpe_bias.reshape(B, self.n_heads, 2 * H - 1, 2 * W - 1),
            grid=displacement[..., (1, 0)],
            mode='bilinear', align_corners=True
        )  # B * g, h_g, HW, Ns

        attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)

        attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        out = torch.einsum('b m n, b c n -> b c m', attn, v)


        out = out.reshape(B, C, H, W)#1 1024 7 7

        y = self.conv(out)# 1 16 7 7
        return y, pos, reference



class TransformerStage(nn.Module):

    def __init__(self, fmap_size,
                 dim_in, dim_embed, heads):

        super().__init__()
        fmap_size = to_2tuple(fmap_size)
        hc = dim_embed // heads
        assert dim_embed == heads * hc
        self.proj = nn.Conv2d(dim_in, dim_embed, 1, 1, 0) if dim_in != dim_embed else nn.Identity()
        #self.norm=nn.BatchNorm2d(1024, eps=NORM_EPS)

        self.layer_norms = LayerNormProxy(dim_embed)
        self.attns = DAttentionBaseline(fmap_size, fmap_size, heads,hc)

    def forward(self, x):

        x = self.proj(x)

        positions = []
        references = []
        #x, pos, ref = self.attns[d](self.layer_norms[2 * d](x))
        x=self.layer_norms(x)
        x, pos, ref = self.attns(x)
        positions.append(pos)
        references.append(ref)
        return x, positions, references


class DAT(nn.Module):
    def __init__(self,dims=1024,heads=32):
        super().__init__()
        image_size = 7
        dim1 = dims
        dim2 = dims
        self.stages = TransformerStage(image_size, dim1, dim2,heads)
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

    def forward(self, x):
        x, pos, ref = self.stages(x)
        return x


if __name__ == '__main__':
    x = torch.randn(1, 1024, 7, 7)
    model = DAT()
    #print(model)
    x = model(x)
    print(x.shape)

