import math
import pandas as pd
import Quandl
import numpy as np
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression

df = Quandl.get("WIKI/GOOGL")

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_CHG'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close','HL_PCT','PCT_CHG','Adj. Volume']]
#print df.head()

forecast_col = 'Adj. Close'

df.fillna(-99999, inplace=True) # have to replace the NaN data with something

forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])
X = preprocessing.scale(X) #keeps features within range of -1 to 1, supposed to speed up processing?
y = np.array(df['label'])
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
clf = svm.SVR()
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print(confidence)
clf = LinearRegression(n_jobs = -1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print confidence
