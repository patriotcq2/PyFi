import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader as web


style.use('ggplot')
df=pd.read_csv('D:/TSLA.csv',parse_dates=True,index_col=0)
df['100ma']=df['Adj Close'].rolling(window=100,min_periods=0).mean()
df.dropna(inplace=True)

print(df.head())