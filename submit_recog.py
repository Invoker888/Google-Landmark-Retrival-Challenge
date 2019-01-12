#-*- coding:utf-8 -*-
import torch as t
import numpy as np
import pandas as pd

result = t.load('result.pth')
df = pd.DataFrame.from_dict(result,orient='index')
df = df.reset_index() #添加索引
df.columns = ['id', 'landmarks']
df['id'] = df['id'].map(lambda x: x[0:-4])
test = pd.read_csv('../kaggle-google-recognization/test.csv')
add_df = test[~test['id'].isin(df['id'])]
add_df['url'] = '' #未被下载的图片全部识别为non landmarks
add_df.columns = ['id', 'landmarks']
df = pd.concat([df,add_df])
df.to_csv('result0424.csv', index=False)
