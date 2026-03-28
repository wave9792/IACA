import os
import json
import random
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from config_cropping import cfg


IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]


def rescale_bbox(bbox, ratio_w, ratio_h):
    bbox = np.array(bbox).reshape(-1, 4)
    bbox[:, 0] = np.floor(bbox[:, 0] * ratio_w)
    bbox[:, 1] = np.floor(bbox[:, 1] * ratio_h)
    bbox[:, 2] = np.ceil(bbox[:, 2] * ratio_w)
    bbox[:, 3] = np.ceil(bbox[:, 3] * ratio_h)
    return bbox.astype(np.float32)


def get_target_size(im_width, im_height, keep_aspect_ratio):
    if keep_aspect_ratio:
        scale = float(cfg.image_size[0]) / min(im_height, im_width)
        h = round(im_height * scale / 32.0) * 32
        w = round(im_width * scale / 32.0) * 32
    else:
        h = cfg.image_size[1]
        w = cfg.image_size[0]
    return int(w), int(h)


class FCDBDataset(Dataset):
    def __init__(self, split, keep_aspect_ratio=False):
        self.split = split
        self.keep_aspect = keep_aspect_ratio
        self.data_dir = cfg.FCDB_dir
        assert os.path.exists(self.data_dir), self.data_dir

        self.image_dir = os.path.join(self.data_dir, 'data')
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

        self.augment_transformer = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
        ])

    def parse_annotations(self, split):
        if split == 'train':
            split_file = os.path.join(self.data_dir, 'cropping_training_set.json')
        else:
            split_file = os.path.join(self.data_dir, 'cropping_testing_set.json')

        assert os.path.exists(split_file), split_file

        with open(split_file, 'r', encoding='utf-8') as f:
            origin_data = json.load(f)

        annos = {}
        for item in origin_data:
            url = item['url']
            image_name = os.path.split(url)[-1]
            image_path = os.path.join(self.image_dir, image_name)

            if os.path.exists(image_path):
                x, y, w, h = item['crop']
                crop = [x, y, x + w, y + h]
                annos[image_name] = crop

        print('{} set, {} images'.format(split, len(annos)))
        return annos

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_file = os.path.join(self.image_dir, image_name)

        with Image.open(image_file) as img:
            image = img.convert('RGB')

        im_width, im_height = image.size

        w, h = get_target_size(im_width, im_height, self.keep_aspect)

        resized_image = image.resize((w, h), Image.Resampling.LANCZOS)

        # 裁剪框保留原图坐标；训练时在 train_image_cropping.py 中再映射到输入分辨率
        crop = torch.tensor(self.annos[image_name], dtype=torch.float32)

        if self.data_augment:
            if random.uniform(0, 1) > 0.5:
                resized_image = ImageOps.mirror(resized_image)
                temp_x1 = crop[0].clone()
                crop[0] = im_width - crop[2] - 1
                crop[2] = im_width - temp_x1 - 1

            resized_image = self.PhotometricDistort(resized_image)

        view1 = self.image_transformer(resized_image)

        if self.split == 'train':
            view2 = self.augment_transformer(resized_image)
            return view1, view2, crop, im_width, im_height

        return view1, crop, im_width, im_height, image_file


class FLMSDataset(Dataset):
    def __init__(self, split='test', keep_aspect_ratio=False):
        self.split = split
        self.keep_aspect = keep_aspect_ratio
        self.data_dir = cfg.FLMS_dir
        assert os.path.exists(self.data_dir), self.data_dir

        self.image_dir = os.path.join(self.data_dir, 'image')
        assert os.path.exists(self.image_dir), self.image_dir

        self.annos = self.parse_annotations()

        self.image_list = sorted(list(self.annos.keys()))

        self.image_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
        ])

    def parse_annotations(self):
        image_crops_file = os.path.join(self.data_dir, '500_image_dataset.mat')
        assert os.path.exists(image_crops_file), image_crops_file

        import scipy.io as scio

        image_crops = {}
        anno = scio.loadmat(image_crops_file)

        for i in range(anno['img_gt'].shape[0]):
            image_name = anno['img_gt'][i, 0][0][0]
            gt_crops = anno['img_gt'][i, 0][1]
            gt_crops = gt_crops[:, [1, 0, 3, 2]]
            keep_index = np.where((gt_crops < 0).sum(1) == 0)
            gt_crops = gt_crops[keep_index].tolist()
            image_crops[image_name] = gt_crops

        print('{} images'.format(len(image_crops)))
        return image_crops

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_file = os.path.join(self.image_dir, image_name)

        with Image.open(image_file) as img:
            image = img.convert('RGB')

        im_width, im_height = image.size

        w, h = get_target_size(im_width, im_height, self.keep_aspect)

        resized_image = image.resize((w, h), Image.Resampling.LANCZOS)
        im = self.image_transformer(resized_image)

        crop = torch.tensor(np.array(self.annos[image_name]).reshape(-1, 4), dtype=torch.float32)

        return im, crop, im_width, im_height, image_file


if __name__ == '__main__':
    print('===== check FCDB train split =====')
    fcdb_trainset = FCDBDataset(split='train')
    trainloader = DataLoader(fcdb_trainset, batch_size=1, num_workers=1)
    for batch_idx, data in enumerate(trainloader):
        view1, view2, crop, im_width, im_height = data
        print(view1.shape, view2.shape, crop.shape, im_width, im_height)
        if batch_idx == 0:
            break

    print('===== check FCDB test split =====')
    fcdb_testset = FCDBDataset(split='test')
    testloader = DataLoader(fcdb_testset, batch_size=1, num_workers=1)
    for batch_idx, data in enumerate(testloader):
        im, crop, im_width, im_height, image_file = data
        print(image_file[0])
        print(im.shape, crop.shape, im_width, im_height)
        if batch_idx == 0:
            break

    print('===== check FLMS test split =====')
    flms_testset = FLMSDataset(split='test')
    flms_loader = DataLoader(flms_testset, batch_size=1, num_workers=1)
    for batch_idx, data in enumerate(flms_loader):
        im, crop, w, h, file = data
        print(file[0])
        print(im.shape, crop.shape, w, h)
        if batch_idx == 0:
            break