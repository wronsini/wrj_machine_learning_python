import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing

df = pd.read_csv('./static/teleCust1000t.csv')
print(df.head(10))
print(df['custcat'].value_counts())
df.hist(column='income', bins=50)
plt.show()
print(df.columns)
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
print(X[0:5])
y = df['custcat'].values
print(y[0:5])

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
print(X)