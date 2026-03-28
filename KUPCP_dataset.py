import os
import cv2
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from config_classification import cfg


IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]

composition_cls = [
    'rule of thirds(RoT)',
    'vertical',
    'horizontal',
    'diagonal',
    'curved',
    'triangle',
    'center',
    'symmetric',
    'pattern'
]


class CompositionDataset(Dataset):
    def __init__(self, split, keep_aspect_ratio):
        self.split = split
        self.keep_aspect = keep_aspect_ratio
        data_root = cfg.KUPCP_dir
        assert os.path.exists(data_root), data_root

        if split == 'train':
            self.image_dir = os.path.join(data_root, 'train_img')
            self.label_txt = os.path.join(data_root, 'train_label.txt')
        else:
            self.image_dir = os.path.join(data_root, 'test_img')
            self.label_txt = os.path.join(data_root, 'test_label.txt')

        assert os.path.exists(self.image_dir), self.image_dir
        assert os.path.exists(self.label_txt), self.label_txt

        self.annotations = self.gather_annotation()

        if self.split == 'train':
            self.transformer = transforms.Compose([
                transforms.Resize((cfg.image_size[0], cfg.image_size[1])),
                transforms.ColorJitter(
                    brightness=0.125,
                    contrast=0.5,
                    saturation=0.5,
                    hue=0.05
                ),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
            ])
        else:
            self.transformer = transforms.Compose([
                transforms.Resize((cfg.image_size[0], cfg.image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
            ])

    def gather_annotation(self):
        image_list = sorted(
            os.listdir(self.image_dir),
            key=lambda k: float(os.path.splitext(k)[0])
        )

        annotation = []
        image_idx = 0
        txt_idx = 0

        with open(self.label_txt, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                txt_idx += 1

                if image_idx >= len(image_list):
                    break

                image_name = image_list[image_idx]

                if txt_idx < int(os.path.splitext(image_name)[0]):
                    continue

                image_idx += 1

                assert int(os.path.splitext(image_name)[0]) == txt_idx, [txt_idx, image_name, line]

                image_path = os.path.join(self.image_dir, image_name)
                assert os.path.exists(image_path), image_path

                labels = [int(l) for l in line.strip().split(' ')]
                assert len(labels) == len(composition_cls), labels

                categories = [i for i in range(len(labels)) if labels[i] == 1]

                if len(categories) > 0:
                    if self.split == 'test':
                        annotation.append((image_name, categories))
                    else:
                        for c in categories:
                            annotation.append((image_name, [c]))

        print('{} set, total {} images'.format(self.split, len(annotation)))
        return annotation

    def gather_training_annotation(self):
        image_list = sorted(
            os.listdir(self.image_dir),
            key=lambda k: float(os.path.splitext(k)[0])
        )

        annotation = [[] for _ in range(len(composition_cls))]

        with open(self.label_txt, 'r', encoding='utf-8') as f:
            for image_name, line in zip(image_list, f.readlines()):
                image_path = os.path.join(self.image_dir, image_name)
                assert os.path.exists(image_path), image_path

                labels = [int(l) for l in line.strip().split(' ')]
                assert len(labels) == len(composition_cls), labels

                categories = [i for i in range(len(labels)) if labels[i] == 1]
                if len(categories) > 0:
                    for c in categories:
                        annotation[c].append(image_name)

        for c in range(len(composition_cls)):
            print('{}, total {} training images'.format(
                composition_cls[c], len(annotation[c]))
            )
        return annotation

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_name, cls = self.annotations[index]

        if self.split == 'train' and len(cls) > 1:
            cls = [cls[-1]]

        cls = torch.tensor(cls, dtype=torch.long)
        image_file = os.path.join(self.image_dir, image_name)

        with Image.open(image_file) as img:
            src = img.convert('RGB')

        im = self.transformer(src)

        return im, cls, image_file


def check_jpg_file(path):
    file_list = [file for file in os.listdir(path) if file.endswith('.jpg')]
    for file in file_list:
        image_file = os.path.join(path, file)
        with open(image_file, 'rb') as f:
            check_chars = f.read()[-2:]
        if check_chars != b'\xff\xd9':
            im = cv2.imread(image_file)
            cv2.imwrite(image_file, im)
        else:
            _ = cv2.imread(image_file)


if __name__ == '__main__':

    comp_dataset = CompositionDataset(split='train', keep_aspect_ratio=False)
    dataloader = DataLoader(comp_dataset, batch_size=8, num_workers=0, shuffle=True)

    for batch_idx, data in enumerate(dataloader):
        im, cls, image_file = data
        print(im.shape, cls.shape, len(image_file))
        if batch_idx == 0:
            break