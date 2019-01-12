#-*- coding:utf-8 -*-
import torchvision as tv
import torch as t
from torch.utils import data
from torch.autograd import Variable
import time
from config import Config
from dataset import TripletImageLoader
from torch.optim import lr_scheduler
import numpy as np
import tqdm
from resnet import resnet50
import pdb

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

def save(model, epoch):
    prefix = './checkpoints/rank_resnet50'
    path = '{prefix}_{time}_{epoch}'.format(prefix = prefix,
                                    time=time.strftime('%m%d_%H%M'), epoch=epoch)
    t.save(model.state_dict(), path)

def load(model, path):
    data = t.load(path)
    model.load_state_dict(data)
    return model

#配置模型
model = resnet50(pretrained=True)
# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model.fc.in_features #获取全连接层的输入channel个数
model.fc = t.nn.Linear(num_ftrs, 14951)
model = t.nn.DataParallel(model)
load_model_path = './checkpoints/rank+instance_resnet50_0424_1120_4'
model = load(model, load_model_path)
model.cuda()

def train():
    #加载数据
    train_data = TripletImageLoader(train=True, instance = True, rank=True)
    dataloader = data.DataLoader(train_data,
                                   batch_size = 64,
                                   shuffle = True,
                                   pin_memory = True,
                                   drop_last = True,
                                   num_workers = 6)

    #损失函数和c优化器
    crossEntropy_loss = t.nn.CrossEntropyLoss()
    triplet_loss = t.nn.TripletMarginLoss(margin=1.0, p=2)
    #optimizer = t.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #optimizer = t.optim.Adam(model.parameters(), lr = 1e-3, weight_decay = opt.weight_decay)
    optimizer = t.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose = True, patience = 1, eps=1e-06)

    #训练
    num_epochs = 5
    for epoch in range(0, num_epochs):
        start = time.time()
        message = 'Epoch {}/{}'.format(epoch, num_epochs)
        print(message)
        print('-' * 10)

        with open('log_file.txt', 'a') as f:
            f.write(message + '\n' + '-' * 10 + '\n')

        epoch_loss = 0.0
        epoch_correct = 0

        for i, (imgs, labels) in tqdm.tqdm(enumerate(dataloader)):
            #imgs1_dim = (batch_size, 3, 224, 224)
            optimizer.zero_grad() #梯度归零
            batch_correct = 0
            pdb.set_trace()
            if len(imgs) != 3:
                imgs1, pos_label = imgs, labels
                x = Variable(imgs).cuda()
                pos_label = Variable(pos_label).cuda()
                _, instance_x1 = model(x)
                loss = crossEntropy_loss(instance_x1, pos_label) 
            else:
                imgs1, imgs2, imgs3 = imgs
                pos_label, neg_label = labels
                x1 = Variable(imgs1).cuda()
                y1 = Variable(imgs2).cuda()
                y2 = Variable(imgs3).cuda()
                
                pos_label = Variable(pos_label).cuda()
                neg_label = Variable(neg_label).cuda()

                triplet_x1, instance_x1 = model(x1)
                triplet_y1, instance_y1 = model(y1)
                triplet_y2, instance_y2 = model(y2)

                rank_loss = triplet_loss(triplet_x1, triplet_y1, triplet_y2)
                if train_data.instance:
                    instance_loss = crossEntropy_loss(instance_x1, pos_label) + crossEntropy_loss(instance_y1, pos_label) + crossEntropy_loss(instance_y2, neg_label)
                    loss = rank_loss + instance_loss
                else:
                    loss = rank_loss
            loss.backward() #一定要写在for循环内部，否则循环内forward产生的中间value一直会保存直到backward
            optimizer.step()

            epoch_loss += loss.data[0]
            score = instance_x1.data.cpu().numpy()
            y_pred = np.argmax(score, 1)  # 每个图像对应的类别
            for j, pred in enumerate(y_pred):
                if pos_label.data[j] == pred:
                    batch_correct += 1
                    epoch_correct += 1

            if i % 10 == 0:
                batch_acc = batch_correct / imgs1.size()[0]
                print(' loss: {:.4f} Acc: {:.4f}'.format(loss.data[0], batch_acc))

        epoch_loss /= len(dataloader)
        epoch_acc = epoch_correct / len(train_data)
        meter = 'Train Loss: {:.4f} Acc: {:.4f}'.format(
            epoch_loss, epoch_acc)
        print(meter)

        scheduler.step(epoch_loss)
        save(model, epoch)
        end = time.time()
        time_message = 'Train time: {:.4f}'.format(end - start)
        print(time_message)
        with open('log_file.txt', 'a') as f:
            f.write(meter + '\n' + time_message + '\n\n')

train()