import numpy as np
import pandas as pd
from collections import Counter
from sklearn.svm import SVC

import  sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier,RandomForestClassifier
from sklearn.svm import LinearSVC
def process_data_for_labels(ticker):
    hm_days = 7
    df = pd.read_csv('sp500.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)
    for i in range(1, hm_days+1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
    df.fillna(0, inplace=True)
    # print(df.head())
    df.to_csv('SP5002.csv')
    return tickers, df

#process_data_for_labels('MMM')

def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0

def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(map( buy_sell_hold,
                                               df['{}_1d'.format(ticker)],
                                               df['{}_2d'.format(ticker)],
                                               df['{}_3d'.format(ticker)],
                                               df['{}_4d'.format(ticker)],
                                               df['{}_5d'.format(ticker)],
                                               df['{}_6d'.format(ticker)],
                                               df['{}_7d'.format(ticker)] ))
    vals = df['{}_target'.format(ticker)].values.tolist()
    #print(df['{}_target'.format(ticker)].values)
    str_vals = [str(i) for i in vals]
    print('Data spread:',Counter(str_vals))
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)
    x=df_vals.values
    y=df['{}_target'.format(ticker)].values
    return x,y,df

def do_ml(ticker):
    x,y,df=extract_featuresets(ticker)
    x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25)
    # clf=KNeighborsClassifier()

    clf=VotingClassifier([('lsvc',LinearSVC()),('knn',KNeighborsClassifier()),('rfor',RandomForestClassifier())])
    clf.fit(x_train,y_train)
    confidence=clf.score(x_test,y_test)
    print('Accuracy:',confidence)
    predictions=clf.predict(x_test)
    print('Predicted Spread:',Counter(predictions))
    return confidence
do_ml('MMM')



