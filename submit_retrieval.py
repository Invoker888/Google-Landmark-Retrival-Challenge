import numpy as np
import pandas as pd
import random
import os
import time

topK = np.load('top100_0505_1545.npy').astype('int') #####TODO
test_path = '/home/supermerry/landmark/kaggle-google-retrivial/resize_test_image/'
index_path = '/home/supermerry/landmark/kaggle-google-retrivial/resize_index_image/'
test_file = os.listdir(test_path)
index_file = os.listdir(index_path)
test_file = [x[:-4] for x in test_file]
index_file = [x[:-4] for x in index_file]

test_df =  pd.read_csv('../kaggle-google-recognization/test.csv') #####TODO

result = []
for i, x in enumerate(topK):
    query = test_file[i]
    documents = ' '.join([index_file[idx] for idx in x])
    result.append([query, documents])
add_df = test_df[~test_df['id'].isin(test_file)]
add_file = list(add_df['id'])

for name in add_file:
    query = name
    documents = result[0][1] # 缺少的行都用第一行数据的查询来代替
    result.append([query, documents])
submit = pd.DataFrame(result)
submit.columns = ['id', 'images']
filename = 'retrieval_result%s.csv'%time.strftime('%m%d_%H%M')
submit.to_csv(filename, index=False)