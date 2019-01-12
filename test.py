#python test.py instance_resnet50_0409_2147_4
import torchvision as tv
import torch as t
from PIL import Image
from torch.utils import data
from torch.autograd import Variable
import time
import os
import numpy as np
from torchvision import transforms as T
import tqdm
from resnet import resnet50
from sys import argv

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

normalize = T.Normalize(mean = [0.485, 0.456, 0.406], 
                        std = [0.229, 0.224, 0.225])

img_path = '/home/supermerry/landmark/kaggle-google-retrivial/resize_test_image/'

class TripletImageLoader(data.Dataset):
    
    def __init__(self):
        '''
        获取所有图像的信息
        '''
        self.all_img = os.listdir(img_path)
        self.transforms = T.Compose([
            T.Scale(224),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize
            ])
    
    def __getitem__(self, index):
        '''
        返回单张图像的数据和label
        '''
        filename = self.all_img[index]
        img = Image.open(img_path + filename)
        img = self.transforms(img)
        return img, filename
    
    def __len__(self):
       '''
       返回测试集中所有图像的个数
       '''
       return len(self.all_img)
            
val_data = TripletImageLoader()
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
    result = {}
    for i, (imgs, filenames) in tqdm.tqdm(enumerate(dataloader)):
        #type of filenames is list
        input = Variable(imgs, volatile=True).cuda()
        score = model(input)
        score = t.nn.Softmax()(score[1])
        score = score.data.cpu().numpy()

        y_pred = np.argmax(score, 1)
        for j, pred in enumerate(y_pred):
            result[filenames[j]] = '{:d} {:.2f}'.format(pred, score[j][pred])
    return result

result = val(model, val_dataloader)
t.save(result, 'result.pth')