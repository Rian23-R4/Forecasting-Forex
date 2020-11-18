import time
import pandas as pd
import datetime
import csv
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service

os.system('cls')
temp = 0
loop = 50
dataSampling = 10
y = 0
for i in range(loop+1):
    i += 1
    print(i)
    f = open('Data %i.csv' % i, 'w')
    writer = csv.writer(f)
    writer.writerow(('Date', 'Nilai'))
    f.close()

f = open(f'Data {loop+1}.csv', 'w')
writer = csv.writer(f)
writer.writerow(('Date', 'Nilai'))

service = Service(executable_path='C:\webdrivers/chromedriver.exe')
service.start()
driver = webdriver.Remote(service.service_url)
driver.get('https://olymptrade.com/platform')


def Write(WW):
    for row in WW:
        for e in row:
            writer2.writerows(e)


entrada = input("Tekan Enter untuk memulai")
while(1):
    Time = str(datetime.datetime.now())[17:19]
    if Time == "00" or Time == "30":
        os.system('cls')
        temp += 1
        if(temp == dataSampling+1):
            f.close()

            for i in range(loop):
                i += 1
                i2 = i + 1
                df = pd.read_csv('Data %i.csv' % i2, index_col=0)
                df.to_csv('Data %i.csv' % i)
            temp = 1
            f = open(f'Data {loop+1}.csv', 'w')
            writer = csv.writer(f)
            writer.writerow(('Date', 'Nilai'))
        print("Jumlah Data : ", temp)
        soup_home = driver.find_elements_by_xpath('/html/body/div/div/div[3]/main/div[2]/div/div/div[1]/div/div/div[1]/div/div[1]/div[1]/div[7]/div/div[1]')
        for item in soup_home:
            hasil = item.text
            y = float(hasil.replace(',', ''))
            print("Nilai Terkini: ", item.text)
        currentDT = datetime.datetime.now()
        s = [currentDT, f'{y}']
        writer.writerow(s)
    time.sleep(1)
