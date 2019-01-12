# coding: utf-8

# In[1]:


import torchvision as tv
import torch as t
from PIL import Image
from torch.utils import data
from torch.autograd import Variable
import time
import os
import numpy as np
from torchvision import transforms as T
from dataset import TripletImageLoader
import tqdm
from resnet import resnet50
from sys import argv

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


val_data = TripletImageLoader(train=False)
val_dataloader = data.DataLoader(val_data,
                                 batch_size = 1024,
                                 shuffle = False,
                                 pin_memory = True,
                                 num_workers = 4)

model = resnet50(pretrained=False)
num_ftrs = model.fc.in_features #获取全连接层的输入channel个数
model.fc = t.nn.Linear(num_ftrs, 14951)
model = t.nn.DataParallel(model)
model.load_state_dict(t.load('./checkpoints/%s'%argv[1]))
model.cuda()

def val(model, dataloader):
    model.eval()
    n_class = 14951
    confusion_matrix = np.zeros((n_class, n_class))
    for i, (imgs, label) in tqdm.tqdm(enumerate(dataloader)):
        input = Variable(imgs, volatile=True).cuda()
        score = model(input)
        score = score[1].data.cpu().numpy()

        y_pred = np.argmax(score, 1)
        for j, pred in enumerate(y_pred):
            confusion_matrix[label[j], pred] += 1
    return confusion_matrix

start = time.time()
confusion_matrix = val(model, val_dataloader)
end = time.time()
print (end - start)
with open('test.txt', 'a') as f:
    f.write('val time: %f\n'%(end - start))

# In[13]:

correct= 0
for i in range(14951):
    correct += confusion_matrix[i,i]
acc = correct / (np.sum(confusion_matrix))
print (acc)


# In[17]:

with open('test.txt', 'a') as f:
    f.write('val acc: %f'%acc)

t.save(confusion_matrix, 'resnet50_confusion_matrix%s.pth'%argv[2])
