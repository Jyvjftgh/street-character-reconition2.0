import os
import glob
import json
import pandas as pd
import torchvision.transforms as T
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from config import data_dir, config

# ---------------- 辅助统计函数 ----------------
def data_summary():
    train_list = glob.glob(data_dir['train_data'] + '*.png')
    test_list = glob.glob(data_dir['test_data'] + '*.png')
    val_list = glob.glob(data_dir['val_data'] + '*.png')
    print('Train image counts: %d' % len(train_list))
    print('Val image counts: %d' % len(val_list))
    print('Test image counts: %d' % len(test_list))


def look_train_json():
    with open(data_dir['train_label'], 'r', encoding='utf-8') as f:
        content = f.read()
    content = json.loads(content)
    print(content['000000.png'])


def look_submit():
    df = pd.read_csv(data_dir['submit_file'], sep=',')
    print(df.head(5))


def label_summary():
    marks = json.load(open(data_dir['train_label'], 'r'))
    dicts = {}
    for img, mark in marks.items():
        num_digits = len(mark['label'])
        if num_digits not in dicts:
            dicts[num_digits] = 0
        dicts[num_digits] += 1
    dicts = sorted(dicts.items(), key=lambda x: x[0])
    for k, v in dicts:
        print(f'Number of images with {k} digits: {v}')


# ---------------- 数据增强类 ----------------
class AdvancedTransforms:
    @staticmethod
    def get_train_transforms():
        return T.Compose([
            T.Resize((config.img_size, config.img_size)),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.RandomGrayscale(p=0.2),
            T.RandomAffine(10, translate=(0.15, 0.15), scale=(0.8, 1.2), shear=10),
            T.RandomHorizontalFlip(p=0.5),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
            T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ])

    @staticmethod
    def get_val_transforms():
        return T.Compose([
            T.Resize((config.img_size, config.img_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ])
