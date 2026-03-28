import os
import glob
import cv2
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
image_size = (224, 224)

theme_cls = ['orange', 'art', 'beach', 'bird', 'black', 'blue', 'bridge', 'car', 'cat',
             'city', 'citysky', 'clouds', 'dog', 'family', 'flower', 'food', 'garden',
             'green', 'grey', 'holiday', 'house', 'lake', 'light', 'macro', 'moon',
             'music', 'nature', 'night', 'old', 'park', 'people', 'pink', 'portrait',
             'river', 'scene', 'sea', 'sky', 'snow', 'street', 'summer', 'sun',
             'sunset', 'tree', 'water', 'white', 'winter', 'yellow']


from paths import TAD66K_DIR

DATA_ROOT = TAD66K_DIR
csv_path = os.path.join(str(DATA_ROOT), 'labels', 'labels', 'unmerge')
theme_image_path = os.path.join(str(DATA_ROOT), 'TAD66K_dataset')

class ThemeDataset(Dataset):
    def __init__(self, split, keep_aspect_ratio):
        self.split = split
        self.keep_aspect = keep_aspect_ratio

        assert os.path.exists(csv_path), csv_path
        assert os.path.exists(theme_image_path), theme_image_path

        if split == 'train':
            self.image_dir = theme_image_path
            self.path_to_csv = os.path.join(csv_path, 'train')
        else:
            self.image_dir = theme_image_path
            self.path_to_csv = os.path.join(csv_path, 'test')

        assert os.path.exists(self.path_to_csv), self.path_to_csv
        self.annotations = self.gather_annotation()

        if self.split == 'train':
            self.transformer = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
            ])
        else:
            self.transformer = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
            ])

    def gather_annotation(self):
        annotation = []

        files = sorted(glob.glob(os.path.join(self.path_to_csv, '*.csv')))
        assert len(files) > 0, f'No csv files found in {self.path_to_csv}'

        for path in files:
            theme_name = os.path.splitext(os.path.basename(path))[0]

            if theme_name not in theme_cls:
                print(f'[Warning] Skip unknown theme csv: {theme_name}')
                continue

            theme_idx = theme_cls.index(theme_name)
            df = pd.read_csv(path)

            if 'image' not in df.columns:
                raise KeyError(f'"image" column not found in {path}')

            for index in range(len(df)):
                row = df.iloc[index]
                image_name = row['image']
                image_file = os.path.join(self.image_dir, image_name)

                if os.path.exists(image_file):
                    annotation.append((image_name, [theme_idx]))

        print('{} set, total {} images'.format(self.split, len(annotation)))
        return annotation

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_name, theme = self.annotations[index]
        theme = torch.tensor(theme).long()
        image_file = os.path.join(self.image_dir, image_name)
        src = Image.open(image_file).convert('RGB')
        im = self.transformer(src)
        return im, theme, image_file


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
    theme_dataset = ThemeDataset(split='test', keep_aspect_ratio=False)
    dataloader = DataLoader(theme_dataset, batch_size=8, num_workers=0, shuffle=True)

    for batch_idx, data in enumerate(dataloader):
        im, cls, comp = data
        print(im.shape, cls, len(comp))
        if batch_idx == 0:
            break