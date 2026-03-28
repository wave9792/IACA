import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from GRRM import DAT
from nextvit_cls import NextViT
from config_cropping import cfg
from paths import NEXTVIT_PRETRAIN


class Next_Vit(nn.Module):
    def __init__(self, loadweights=True):
        super(Next_Vit, self).__init__()
        self.model = NextViT()

        if loadweights:
            weight_filepath_vit = str(NEXTVIT_PRETRAIN)
            weights_dict_vit = torch.load(weight_filepath_vit, map_location='cpu')['model']
            print(self.model.load_state_dict(weights_dict_vit, strict=False))

    def forward(self, x):
        return self.model(x)


class DAT_cropping(nn.Module):
    def __init__(self):
        super(DAT_cropping, self).__init__()
        self.model = DAT()

    def forward(self, x):
        return self.model(x)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.projection(x)


class CompositionModel(nn.Module):
    def __init__(self):
        super(CompositionModel, self).__init__()
        self.comp_types = 9
        self.conv1 = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.GAP = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1)
        )
        self.fc_layer = nn.Linear(128, self.comp_types, bias=True)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        gap = self.GAP(x)
        logits = self.fc_layer(gap)
        conf = F.softmax(logits, dim=1)

        with torch.no_grad():
            b, c, h, w = x.shape
            weight = self.fc_layer.weight.data
            trans_w = einops.repeat(weight, 'n c -> b n c', b=b)
            trans_x = einops.rearrange(x, 'b c h w -> b c (h w)')
            cam = torch.matmul(trans_w, trans_x)
            cam = cam - cam.min(dim=-1)[0].unsqueeze(-1)
            cam = cam / (cam.max(dim=-1)[0].unsqueeze(-1) + 1e-12)
            cam = einops.rearrange(cam, 'b n (h w) -> b n h w', h=h, w=w)
            kcm = torch.sum(conf[:, :, None, None] * cam, dim=1, keepdim=True)
            kcm = F.interpolate(kcm, scale_factor=32, mode='bilinear', align_corners=True)

        return logits, kcm


class ThemeModel(nn.Module):
    def __init__(self):
        super(ThemeModel, self).__init__()
        self.theme_types = 47
        self.conv1 = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.GAP = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1)
        )
        self.fc_layer = nn.Linear(128, self.theme_types, bias=True)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        gap = self.GAP(x)
        logits = self.fc_layer(gap)
        conf = F.softmax(logits, dim=1)

        with torch.no_grad():
            b, c, h, w = x.shape
            weight = self.fc_layer.weight.data
            trans_w = einops.repeat(weight, 'n c -> b n c', b=b)
            trans_x = einops.rearrange(x, 'b c h w -> b c (h w)')
            cam = torch.matmul(trans_w, trans_x)
            cam = cam - cam.min(dim=-1)[0].unsqueeze(-1)
            cam = cam / (cam.max(dim=-1)[0].unsqueeze(-1) + 1e-12)
            cam = einops.rearrange(cam, 'b n (h w) -> b n h w', h=h, w=w)
            ktm = torch.sum(conf[:, :, None, None] * cam, dim=1, keepdim=True)
            ktm = F.interpolate(ktm, scale_factor=32, mode='bilinear', align_corners=True)

        return logits, ktm


class CroppingModel(nn.Module):
    def __init__(self, anchor_stride):
        super(CroppingModel, self).__init__()
        self.anchor_stride = anchor_stride
        self.aot = DAT_cropping()

        self.projection = nn.Sequential(
            nn.LazyLinear(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128)
        )

    def forward(self, x, return_embedding=False):
        if return_embedding:
            pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
            emb = self.projection(pooled)
            return emb

        offsets = self.aot(x)
        return offsets


def pos_chchors(image_size, stride):
    shift_w = torch.arange(2, image_size[0], stride)
    shift_h = torch.arange(2, image_size[1], stride)
    shift_w, shift_h = torch.meshgrid(shift_w, shift_h, indexing='ij')
    shifts = torch.stack([shift_w, shift_h], dim=-1)
    all_anchors = einops.rearrange(shifts, 'h w c -> 1 h w c')
    return all_anchors


class PostProcess(nn.Module):
    def __init__(self, anchor_stride, image_size):
        super(PostProcess, self).__init__()
        self.num_anchors = (32 // anchor_stride) ** 2
        all_anchors = pos_chchors(image_size, anchor_stride)
        self.register_buffer('all_anchors', all_anchors)

        grid_x = (all_anchors[..., 0] - image_size[0] / 2) / (image_size[0] / 2)
        grid_y = (all_anchors[..., 1] - image_size[1] / 2) / (image_size[1] / 2)
        grid = torch.stack([grid_x, grid_y], dim=-1)
        self.register_buffer('grid', grid)

    def forward(self, offsets, attribute_map):
        offsets = einops.rearrange(offsets, 'b (n c) h w -> b n h w c', n=self.num_anchors, c=4)
        coords = [F.pixel_shuffle(offsets[..., i], upscale_factor=self.num_anchors // 2) for i in range(4)]
        offsets = torch.stack(coords, dim=-1).squeeze(1)
        regression = torch.zeros_like(offsets)

        regression[..., 0::2] = offsets[..., 0::2] + self.all_anchors[..., 0:1]
        regression[..., 1::2] = offsets[..., 1::2] + self.all_anchors[..., 1:2]

        trans_grid = einops.repeat(self.grid, '1 h w c -> b h w c', b=offsets.shape[0])
        sample_map = F.grid_sample(attribute_map, trans_grid, mode='bilinear', align_corners=True)
        reg_weight = F.softmax(sample_map.flatten(1), dim=1).unsqueeze(-1)

        regression = einops.rearrange(regression, 'b h w c -> b (h w) c')
        weighted_reg = torch.sum(reg_weight * regression, dim=1)
        return weighted_reg


class CACNet(nn.Module):
    def __init__(self, loadweights=True):
        super(CACNet, self).__init__()
        anchor_stride = 16
        image_size = cfg.image_size

        self.nextvit = Next_Vit(loadweights=loadweights)
        self.composition_module = CompositionModel()
        self.theme_module = ThemeModel()
        self.cropping_module = CroppingModel(anchor_stride)
        self.post_process = PostProcess(anchor_stride, image_size)

    def extract_features(self, im):
        return self.nextvit(im)

    def forward(self, im, operation='cropping'):
        feature = self.extract_features(im)
        return self.forward_from_features(feature, operation=operation)

    def forward_from_features(self, feature, operation='cropping'):
        if operation == 'composition':
            return self.composition_module(feature)

        if operation == 'theme':
            return self.theme_module(feature)

        if operation == 'embedding':
            return self.cropping_module(feature, return_embedding=True)

        offsets = self.cropping_module(feature)
        logits_com, kcm = self.composition_module(feature)
        logits_theme, ktm = self.theme_module(feature)
        mapping = 0.5 * kcm + 0.5 * ktm
        box = self.post_process(offsets, mapping)
        return logits_com, logits_theme, box

    def get_embedding(self, im):
        feature = self.extract_features(im)
        return self.cropping_module(feature, return_embedding=True)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(5, 3, cfg.image_size[0], cfg.image_size[1]).to(device)
    model = CACNet(loadweights=True).to(device)

    cls, theme, box = model(x)
    print(cls.shape, theme.shape, box.shape)

    emb = model.get_embedding(x)
    print('Embedding shape:', emb.shape)