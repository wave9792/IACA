# -*- coding: utf-8 -*-
import os
import random
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from config_cropping import cfg
from paths import GAIC_DIR


IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]


def get_target_size(im_width, im_height, keep_aspect_ratio):
    if keep_aspect_ratio:
        scale = float(cfg.image_size[0]) / min(im_height, im_width)
        h = round(im_height * scale / 32.0) * 32
        w = round(im_width * scale / 32.0) * 32
    else:
        h = cfg.image_size[1]
        w = cfg.image_size[0]
    return int(w), int(h)


class GAICDataset(Dataset):
    def __init__(self, split, keep_aspect_ratio=False):
        self.split = split
        self.keep_aspect = keep_aspect_ratio
        self.data_dir = str(GAIC_DIR)
        assert os.path.exists(self.data_dir), self.data_dir
        self.image_dir = os.path.join(self.data_dir, 'images')
        assert os.path.exists(self.image_dir), self.image_dir
        self.annos = self.parse_annotations(split)
        self.image_list = sorted(list(self.annos.keys()))
        self.data_augment = (cfg.data_augmentation and self.split == 'train')

        self.PhotometricDistort = transforms.ColorJitter(
            brightness=0.125,
            contrast=0.5,
            saturation=0.5,
            hue=0.05
        )

        self.image_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
        ])

    def parse_annotations(self, split):
        if split == 'train':
            split_file = os.path.join(self.data_dir, 'annotations', 'train')
            image_lists = os.listdir(os.path.join(self.data_dir, 'images', 'train'))
        else:
            split_file = os.path.join(self.data_dir, 'annotations', 'test')
            image_lists = os.listdir(os.path.join(self.data_dir, 'images', 'test'))

        assert os.path.exists(split_file), split_file

        annos = {}
        for item in sorted(image_lists):
            image_name = os.path.split(item)[-1]
            anno_name = image_name[:-3] + 'txt'
            anno_path = os.path.join(split_file, anno_name)
            crop = []

            if os.path.exists(anno_path):
                with open(anno_path, 'r', encoding='utf-8') as fid:
                    annotations_txt = fid.readlines()

                for annotation in annotations_txt:
                    annotation_split = annotation.split()
                    crop.append([
                        float(annotation_split[1]),
                        float(annotation_split[0]),
                        float(annotation_split[3]),
                        float(annotation_split[2])
                    ])

            annos[image_name] = crop

        print('{} set, {} images'.format(split, len(annos)))
        return annos

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_file = os.path.join(self.image_dir, self.split, image_name)

        with Image.open(image_file) as img:
            image = img.convert('RGB')

        im_width, im_height = image.size
        w, h = get_target_size(im_width, im_height, self.keep_aspect)
        resized_image = image.resize((w, h), Image.Resampling.LANCZOS)

        crop = np.array(self.annos[image_name]).reshape(-1, 4).astype(np.float32)

        if self.data_augment:
            if random.uniform(0, 1) > 0.5:
                resized_image = ImageOps.mirror(resized_image)
                temp_x1 = crop[:, 0].copy()
                crop[:, 0] = im_width - crop[:, 2]
                crop[:, 2] = im_width - temp_x1
            resized_image = self.PhotometricDistort(resized_image)

        im = self.image_transformer(resized_image)
        return im, crop, im_width, im_height, image_file


# ---- GAICD Ranking Dataset: loads MOS scores and returns scored crop pairs ----

MOS_MEAN = 2.95
MOS_STD = 0.80


class GAICDRankingDataset(Dataset):
    """Loads GAICD images with scored candidate crops for ranking loss.

    Each __getitem__ returns:
        im:          (3, 224, 224)  normalized tensor
        good_crop:   (4,)  [x1, y1, x2, y2] high-score crop box in original coords
        bad_crop:    (4,)  [x1, y1, x2, y2] low-score crop box in original coords
        good_score:  float  raw MOS score of good_crop
        bad_score:   float  raw MOS score of bad_crop
        im_width, im_height: original image size
    """
    def __init__(self, split, keep_aspect_ratio=False):
        self.split = split
        self.keep_aspect = keep_aspect_ratio
        self.data_dir = str(GAIC_DIR)
        assert os.path.exists(self.data_dir), self.data_dir
        self.image_dir = os.path.join(self.data_dir, 'images')
        assert os.path.exists(self.image_dir), self.image_dir
        self.annos = self.parse_annotations_with_scores(split)
        self.image_list = sorted(list(self.annos.keys()))
        self.data_augment = (cfg.data_augmentation and self.split == 'train')

        self.PhotometricDistort = transforms.ColorJitter(
            brightness=0.125,
            contrast=0.5,
            saturation=0.5,
            hue=0.05
        )
        self.image_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
        ])

    def parse_annotations_with_scores(self, split):
        if split == 'train':
            split_file = os.path.join(self.data_dir, 'annotations', 'train')
            image_lists = os.listdir(os.path.join(self.data_dir, 'images', 'train'))
        else:
            split_file = os.path.join(self.data_dir, 'annotations', 'test')
            image_lists = os.listdir(os.path.join(self.data_dir, 'images', 'test'))

        assert os.path.exists(split_file), split_file

        annos = {}
        for item in sorted(image_lists):
            image_name = os.path.split(item)[-1]
            anno_name = image_name[:-3] + 'txt'
            anno_path = os.path.join(split_file, anno_name)

            if os.path.exists(anno_path):
                with open(anno_path, 'r', encoding='utf-8') as fid:
                    annotations_txt = fid.readlines()

                crops_with_scores = []
                for annotation in annotations_txt:
                    annotation_split = annotation.split()
                    if len(annotation_split) < 5:
                        continue
                    score = float(annotation_split[4])
                    if score == -2:  # skip invalid scores
                        continue
                    # annotation format: [x1, y1, x2, y2, score]
                    # stored as: [x1, y1, x2, y2, score]
                    crops_with_scores.append([
                        float(annotation_split[1]),  # x1
                        float(annotation_split[0]),  # y1
                        float(annotation_split[3]),  # x2
                        float(annotation_split[2]),  # y2
                        score
                    ])

                if len(crops_with_scores) >= 2:
                    annos[image_name] = crops_with_scores

        print('GAICD Ranking {} set, {} images with scored crops'.format(split, len(annos)))
        return annos

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_file = os.path.join(self.image_dir, self.split, image_name)

        with Image.open(image_file) as img:
            image = img.convert('RGB')

        im_width, im_height = image.size
        w, h = get_target_size(im_width, im_height, self.keep_aspect)
        resized_image = image.resize((w, h), Image.Resampling.LANCZOS)

        crops_with_scores = self.annos[image_name]
        # Sort by score descending, pick top and bottom
        sorted_crops = sorted(crops_with_scores, key=lambda x: x[4], reverse=True)

        # Pick the best crop and a random poor crop from the lower half
        good_entry = sorted_crops[0]
        num_candidates = max(2, len(sorted_crops))
        low_start = max(1, num_candidates // 2)
        bad_entry = sorted_crops[random.randint(low_start, num_candidates - 1)]

        good_crop = np.array(good_entry[:4], dtype=np.float32)
        bad_crop = np.array(bad_entry[:4], dtype=np.float32)
        good_score = good_entry[4]
        bad_score = bad_entry[4]

        if self.data_augment:
            if random.uniform(0, 1) > 0.5:
                resized_image = ImageOps.mirror(resized_image)
                good_crop[0], good_crop[2] = im_width - good_crop[2], im_width - good_crop[0]
                bad_crop[0], bad_crop[2] = im_width - bad_crop[2], im_width - bad_crop[0]
            resized_image = self.PhotometricDistort(resized_image)

        im = self.image_transformer(resized_image)

        return im, good_crop, bad_crop, good_score, bad_score, im_width, im_height


if __name__ == '__main__':
    print('=' * 5, 'GAICD Ranking Train', '=' * 5)
    gaic_rank_train = GAICDRankingDataset(split='train')
    train_loader = DataLoader(gaic_rank_train, batch_size=2, num_workers=0)
    for batch_idx, data in enumerate(train_loader):
        im, good_crop, bad_crop, good_score, bad_score, im_w, im_h = data
        print('im:', im.shape, 'good_crop:', good_crop, 'bad_crop:', bad_crop)
        print('good_score:', good_score, 'bad_score:', bad_score)
        if batch_idx == 1:
            break
