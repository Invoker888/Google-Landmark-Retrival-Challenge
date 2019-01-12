import os
import torch
import random
import pandas as pd

path = '../kaggle-google-recognization/train.csv'
df = pd.read_csv(path)
img2url = dict(zip(df['id'], df['url']))
nb_class = len(set(df['landmark_id']))
label2img = [[] for _ in range(nb_class)]
img2label = {}
for i in range(len(df)):
    img = df['id'][i]
    if os.path.exists
    landmark = df['landmark_id'][i]
    img2label[img] = landmark 
    label2img[landmark].append(img)

train_img = []; val_img = []
for i in range(nb_class):
    imgs = label2img[i]
    if len(imgs) >= 3:
        random.shuffle(imgs)
        split_idx = int(len(imgs)*0.8)
        train_img += imgs[:split_idx]
        val_img += imgs[split_idx:]
random.shuffle(train_img)
random.shuffle(val_img)
train_label = [img2label[x] for x in train_img]
val_label = [img2label[x] for x in val_img]

dataset_info = {'train_img': train_img,
                'train_label': train_label,
                'val_img': val_img,
                'val_label': val_label,
                'img2url': img2url,
                'label2img': label2img,
                'img2label': img2label}
torch.save(dataset_info, 'dataset_info.pth')