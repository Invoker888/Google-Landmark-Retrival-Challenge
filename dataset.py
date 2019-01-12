#index.csv: 5183张图片丢失 test.csv: 1782 train.csv: 5821
from torch.utils import data
import torch as t
import torchvision as tv
from torchvision import transforms as T
from getTriplet import get_triplet
from PIL import Image
import os

normalize = T.Normalize(mean = [0.485, 0.456, 0.406], 
                        std = [0.229, 0.224, 0.225])

img_path = '/home/supermerry/landmark/kaggle-google-recognization/resize_train_image/'

class TripletImageLoader(data.Dataset):
    
    def __init__(self, train=True, instance = True, rank=False):
        '''
        获取所有视频信息
        '''
        dataset_info = t.load('dataset_info.pth')
        self.img2label = dataset_info['img2label'] #type: dict
        self.label2img = dataset_info['label2img'] #type: list
        self.train = train
        self.rank = rank
        self.instance = instance
        
        if train:
            self.all_img = dataset_info['train_img']
            self.label = dataset_info['train_label']
        else:
            self.all_img = dataset_info['val_img']
            self.label = dataset_info['val_label']

        # 测试集和验证集不用数据增强
        if not train:
            self.transforms = T.Compose([
                T.Scale(224),
                T.CenterCrop(224),
                T.ToTensor(),
                normalize
            ]) 
        # 训练集需要数据增强
        else: 
            self.transforms = T.Compose([
                T.Scale(256),
                T.RandomSizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
            
    def __getitem__(self, index):
        '''
        返回一个视频的数据（包含10帧图片的数据）和label
        '''
        filename = self.all_img[index]
        if self.train:
            if self.rank:
            #返回的都是PIL对象构成的图像和标签
                imgs, labels = get_triplet(img_path, filename, self.all_img, self.img2label, self.label2img)
                imgs = [self.transforms(x) for x in imgs]
                return (imgs[0], imgs[1], imgs[2]), (labels[0], labels[1])
            else:
                label = self.img2label[filename]
                img = Image.open(img_path + filename + '.jpg')
                img = self.transforms(img)
                return img, label
        else:
            label = self.img2label[filename]
            img = Image.open(img_path + filename + '.jpg')
            img = self.transforms(img)
            return img, label
    
    def __len__(self):
       '''
       返回数据集中所有视频的个数
       '''
       return len(self.all_img)