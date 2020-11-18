import pandas as pd
import pymongo
import pandas as pd
import numpy as np
import dns
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import datetime

df = pd.read_csv('Data 48.csv', index_col=0)

minmax = MinMaxScaler().fit(
    df.iloc[:, 0:1].astype('float32'))  # Close index
df_log = minmax.transform(df.iloc[:, 0:1].astype('float32'))  # Close index
df_log = pd.DataFrame(df_log)

df_train = df_log

date_ori = pd.to_datetime(df.index[:]).tolist()
for i in range(10):
    date_ori.append(date_ori[-1] + timedelta(days=1))
date_ori = pd.Series(date_ori).dt.strftime(date_format='%r').tolist()
date_ori[-5:]

plt.figure(figsize=(10, 10))
plt.plot(df['Nilai'], label='true trend', c='black')
plt.legend()
plt.xticks(date_ori[::2], rotation ='vertical')
plt.savefig('Forecast 1.png')

print(date_ori)
print(type(date_ori))