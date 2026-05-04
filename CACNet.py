# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
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


# ---- Ranking Head: predicts aesthetic quality score from ROI features ----

class RankingHead(nn.Module):
    """Predicts a scalar quality score from ROI-aligned features of a crop region.

    Takes conv features pooled by ROI Align → FC layers → scalar score.
    """
    def __init__(self, input_channels=1024, roi_size=7, hidden_dim=512):
        super(RankingHead, self).__init__()
        self.roi_size = roi_size
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(input_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, roi_features):
        """Args:
            roi_features: (N, C, roi_size, roi_size)  ROI Align output
        Returns:
            scores: (N, 1)  predicted quality score
        """
        x = self.pool(roi_features).flatten(1)
        return self.fc(x)


# ---- Region Projection Head: projects ROI features for region-level contrast ----

class RegionProjectionHead(nn.Module):
    """Projects ROI features to a low-dim embedding for region-level contrastive loss."""
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=128):
        super(RegionProjectionHead, self).__init__()
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, roi_features):
        """Args:
            roi_features: (N, C, roi_size, roi_size)
        Returns:
            embedding: (N, output_dim)
        """
        return self.projection(roi_features)


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

        # 新增：区域级对比学习 + GAICD ranking
        self.ranking_head = RankingHead(input_channels=1024)
        self.region_projection = RegionProjectionHead(input_dim=1024, output_dim=128)

        # 特征图 stride: 224 / 7 = 32
        self.feat_stride = 32.0

    def extract_features(self, im):
        return self.nextvit(im)

    # ---- ROI feature extraction ----

    def extract_roi_features(self, feature_map, boxes_original, im_width, im_height):
        """Extract ROI Align features for crop boxes in original image coordinates.

        Args:
            feature_map:  (B, 1024, 7, 7)
            boxes_original: list of Tensors, each (N_i, 4) in original image coords [x1, y1, x2, y2]
            im_width:  tensor of image widths  (B,)
            im_height: tensor of image heights (B,)

        Returns:
            roi_feats: (total_N, 1024, 7, 7)
        """
        batch_rois = []
        B = feature_map.shape[0]
        _, _, feat_h, feat_w = feature_map.shape

        for i in range(B):
            if boxes_original[i].numel() == 0:
                continue

            box = boxes_original[i]  # (N, 4)

            # Map from original image coords to feature map coords
            scale_x = feat_w / im_width[i].float()
            scale_y = feat_h / im_height[i].float()

            scaled_box = box.clone()
            scaled_box[:, 0] = box[:, 0] * scale_x
            scaled_box[:, 1] = box[:, 1] * scale_y
            scaled_box[:, 2] = box[:, 2] * scale_x
            scaled_box[:, 3] = box[:, 3] * scale_y

            # ROI Align expects [batch_index, x1, y1, x2, y2]
            batch_idx = torch.full((box.shape[0], 1), i, dtype=scaled_box.dtype,
                                   device=scaled_box.device)
            rois = torch.cat([batch_idx, scaled_box], dim=1)
            batch_rois.append(rois)

        if len(batch_rois) == 0:
            return torch.empty(0, feature_map.shape[1], 7, 7,
                               device=feature_map.device, dtype=feature_map.dtype)

        all_rois = torch.cat(batch_rois, dim=0)

        roi_feats = torchvision.ops.roi_align(
            feature_map, all_rois, output_size=7,
            spatial_scale=1.0, aligned=True
        )
        return roi_feats

    # ---- Ranking forward ----

    def forward_ranking(self, feature_map, good_boxes, bad_boxes, im_width, im_height):
        """Predict quality scores for good and bad crop regions.

        Returns:
            score_good: (N, 1)
            score_bad:  (N, 1)
        """
        roi_good = self.extract_roi_features(feature_map, good_boxes, im_width, im_height)
        roi_bad = self.extract_roi_features(feature_map, bad_boxes, im_width, im_height)
        return self.ranking_head(roi_good), self.ranking_head(roi_bad)

    # ---- Region embedding forward ----

    def forward_region_embedding(self, feature_map, boxes, im_width, im_height):
        """Project crop region features to contrastive embedding space."""
        roi_feats = self.extract_roi_features(feature_map, boxes, im_width, im_height)
        return self.region_projection(roi_feats)

    # ---- Main forward ----

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
    print('Composition:', cls.shape, 'Theme:', theme.shape, 'Box:', box.shape)

    emb = model.get_embedding(x)
    print('Embedding shape:', emb.shape)

    # Test ROI feature extraction
    boxes = [torch.tensor([[10., 10., 100., 100.], [50., 50., 200., 200.]], device=device) for _ in range(5)]
    widths = torch.tensor([800.] * 5, device=device)
    heights = torch.tensor([600.] * 5, device=device)
    feat = model.extract_features(x)
    roi = model.extract_roi_features(feat, boxes, widths, heights)
    print('ROI features shape:', roi.shape)

    # Test ranking head
    score = model.ranking_head(roi)
    print('Ranking scores:', score.shape)

    # Test region embedding
    reg_emb = model.region_projection(roi)
    print('Region embedding shape:', reg_emb.shape)
