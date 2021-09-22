import pandas as pd
import os
import glob
import random
import warnings

warnings.filterwarnings('ignore')


df = pd.DataFrame(columns=['anchor', 'positive',
                  'negative', 'person_class', 'label'])

c = 0
for item in os.listdir('trainset'):
    path = os.path.join('trainset', item)
    for item1 in os.listdir(path):
        path1 = os.path.join(path, item1)
        for item2 in os.listdir(path1):
            passport = glob.glob(path1+'\*script*')[0]

            img_dict = {'anchor': passport, 'positive': os.path.join(
                path1, item2), 'person_class': c, 'label': 1}
            df = df.append(img_dict, ignore_index=True)
        c += 1

indexes = []
for i in range(len(df)):
    if df.anchor[i] == df.positive[i] or 'script' in df.positive[i]:
        indexes.append(i)
        
new_df = df.drop(indexes)
new_df.reset_index(drop=True, inplace=True)

for i in range(len(new_df)):
    while True:
        j = random.randint(0, len(new_df)-1)
        if new_df.iloc[i, 2] != new_df.iloc[j, 2]:
            new_df.negative[i] = new_df.iloc[j, 1]
            break
        
new_df=new_df.drop(columns=['person_class']).sample(frac=1)

train, test = new_df[:2198], new_df[2198:]
test.reset_index(drop=True, inplace=True)

train.drop(columns=['label']).to_csv('train_filename.csv', index=False) #saving csv file consisting of columns anchor, positive and negative for training 

for i in range(len(test)):
    c = random.randint(2, 3)
    if c == 2:
        test.label[i] = 1
        test.at[i, 'sample'] = test.at[i, 'positive']
    else:
        test.label[i] = 0
        test.at[i, 'sample'] = test.at[i, 'negative']
        
test[['anchor','sample', 'label']].to_csv('test_filename.csv', index=False)
