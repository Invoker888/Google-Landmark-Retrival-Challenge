#coding:utf8

class Config:
    dataset_info_path = 'dataset_info.pth'
    batch_size = 64
    num_workers = 16
    prefix = './checkpoints/cl_new_resnet50_video'
    model_ckpt = None
    lr = 1e-3
    weight_decay = 1e-4
    max_epoch = 15
    load_model_path = None
