import numpy as np
from PIL import Image
import random
import os

'''
给定一个图像的query，返回triplet中的另外两个输入的图片
需要有的数据结构:类别id到图像的list，图像到类别id的字典
'''
def get_triplet(path, filename, all_img, img2label, label2img):
    pos_label = img2label[filename]
    imgs = label2img[pos_label]
    while True:
        randidx = np.random.randint(0, len(imgs))
        if imgs[randidx] != filename:
            pos_img = imgs[randidx]
            break
    while True:
        randidx = np.random.randint(0, len(all_img))
        img = all_img[randidx]
        if img2label[img] != img2label[filename]:
            neg_img = img
            neg_label = img2label[neg_img]
            break
    query_img = Image.open(path + filename + '.jpg')
    pos_img = Image.open(path + pos_img + '.jpg')
    neg_img = Image.open(path + neg_img + '.jpg')
    
    return (query_img, pos_img, neg_img), (pos_label, neg_label)


