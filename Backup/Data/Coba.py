import warnings
import sys
import tensorflow as tf
import tensorflow.compat.v1 as tf2
tf2.disable_v2_behavior() 
from tensorflow.python.framework import ops
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import os
import time
from datetime import timedelta
from tqdm import tqdm
import msvcrt
import winsound
os.system('cls')

df = pd.read_csv('Data 1.csv', index_col=0)
dataTraining = 50
for i in range(dataTraining-1):
    df = df.append(pd.read_csv(f'Data {i+2}.csv', index_col=0))

print(df)

test_size = int(len(df['Nilai'])*1/10)
minmax = MinMaxScaler().fit(df.iloc[:, 0:1].astype('float32'))  # Close index
df_log = minmax.transform(df.iloc[:, 0:1].astype('float32'))  # Close index
df_log = pd.DataFrame(df_log)

print("")
print(df.index)
date_ori = pd.to_datetime(df.index[-60:]).tolist()
print("")
print(date_ori)
for i in range(test_size):
    date_ori.append(date_ori[-1] + timedelta(days=1))
date_ori = pd.Series(date_ori).dt.strftime(date_format='%r').tolist()
print("")
print(date_ori[:])