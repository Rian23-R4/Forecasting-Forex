import time
import pandas as pd
from datetime import datetime
import csv
import os
import requests
from pymongo import MongoClient

os.system('cls')

try: 
    conn = MongoClient() 
except:   
    print("Could not connect to MongoDB")

for i in range(2):
    i += 1
    f = open('Data %i.csv' % i, 'w')
    writer = csv.writer(f)
    writer.writerow(('Date', 'Nilai'))
    f.close()

f = open(f'Data 2.csv', 'w')
writer = csv.writer(f)
writer.writerow(('Date', 'Nilai'))
URL = "https://www.freeforexapi.com/api/live?pairs=EURUSD"

entrada = input("Tekan Enter untuk memulai")
tamp = requests.get(url = URL).json()['rates']['EURUSD']
dateTamp = datetime.fromtimestamp(tamp['timestamp'])
rate = tamp['rate']
print (f'{dateTamp} : {rate}')
while(1):
    tamp = requests.get(url = URL).json()['rates']['EURUSD']
    date = datetime.fromtimestamp(tamp['timestamp'])
    rate = tamp['rate']
    if dateTamp != date:
        print (f'{date} : {rate}')

    dateTamp = date