#【耗时不到10分钟 python sim.py rank_resnet50_0505_1149_4】
import torchvision as tv
import torch as t
from PIL import Image
from torch.utils import data
from torch.autograd import Variable
import time
import os
import numpy as np
from torchvision import transforms as T
from torch.nn import functional as F
from dataset import TripletImageLoader
import tqdm
from resnet import resnet50
from sys import argv

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3" #####TODO

normalize = T.Normalize(mean = [0.485, 0.456, 0.406], 
                        std = [0.229, 0.224, 0.225])

img_path = '/home/supermerry/landmark/kaggle-google-retrivial/resize_test_image/'#####TODO

class TripletImageLoader(data.Dataset):
    
    def __init__(self):
        '''
        获取所有视频信息
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
        返回一个视频的数据（包含10帧图片的数据）和label
        '''
        filename = self.all_img[index]
        img = Image.open(img_path + filename)
        img = self.transforms(img)
        return img
    
    def __len__(self):
       '''
       返回数据集中所有视频的个数
       '''
       return len(self.all_img)
            
val_data = TripletImageLoader()
val_dataloader = data.DataLoader(val_data,
                                 batch_size = 200, #####TODO
                                 shuffle = False,
                                 pin_memory = True,
                                 num_workers = 4)

feature = np.load('index_feature0505_1452.npy') #####TODO
print('load feature successfully')
result = np.zeros((len(val_data), 100), dtype='float32')

model = resnet50(pretrained=False)
num_ftrs = model.fc.in_features #获取全连接层的输入channel个数
model.fc = t.nn.Linear(num_ftrs, 14951)
model = t.nn.DataParallel(model)
model.load_state_dict(t.load('./checkpoints/%s'%argv[1])) #####TODO
model = model.cuda(0) #####TODO
print ('load model successfully')

def val(model, dataloader):
    model.eval()
    bs = val_dataloader.batch_size
    feat = t.from_numpy(feature)
    feat = F.normalize(feat, p=2, dim=1, eps=1e-12)
    feat = feat.t().cuda(1) #####TODO
    print ('normalize feat successfully')
    
    for i, imgs in tqdm.tqdm(enumerate(dataloader)):
        input = Variable(imgs, volatile=True).cuda(0)#####TODO
        score = model(input) #return a tuple (triplet_out, instance_out)
        s = time.time()
        score = F.normalize(score[0], p=2, dim=1, eps=1e-12)
        e = time.time()
        print ('norm:', e - s)
        score = score.data
        s = time.time()
        similarity = t.matmul(score.cuda(1), feat)#####TODO
        e = time.time()
        print ('sim:', e - s)
        s = time.time()
        topK = similarity.topk(100, dim=1)[1]
        e = time.time()
        print ('topK:', e - s)
        result[i*bs:i*bs+bs] = topK.cpu().numpy()[:,0:100]
    return result

start = time.time()
result = val(model, val_dataloader)
print (result.shape)
end = time.time()
print (end - start)
filename = 'top100_%s.npy'%time.strftime('%m%d_%H%M')
np.save(filename, result)