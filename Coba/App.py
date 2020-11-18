import pymongo
import os
import dns
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
os.system('cls')

client = pymongo.MongoClient(
    "mongodb+srv://Forex23:7bCnUVU09Vlq6TS0@cluster0.fpun3.mongodb.net/forex?retryWrites=true&w=majority")
db = client.forex
collection = db.EURUSD
data = collection.find()
date = []
rate = []
df = []
A = []
B = []
for X in data:
    A.append(X['EURUSD']['rate'])
    B.append(str(X['EURUSD']['date']))
Data = {'ds': B, 'y': A}
df = pd.DataFrame(Data)

model = Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=60, freq='min')
print(df.tail())
print(future.tail())
forecast = model.predict(future)
print(forecast[['ds', 'yhat_lower', 'yhat_upper', 'yhat']])

model.plot(forecast)
plt.savefig('Forecast 1.png')
model.plot_components(forecast)
plt.savefig('Forecast 3.png')

model2 = Prophet()
model2.fit(df[-150:])
model2.plot(forecast[-180:])
plt.savefig('Forecast 2.png')
