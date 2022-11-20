import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

data = pd.read_csv('ADPD.csv')

print(data)

data.describe()

data.info()

print(data.corr())

data=data.drop("Genes",axis=1)
data.shape

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x=data.drop("Disease",axis=1)
# X = pd.DataFrame(sc.fit_transform(x))

import numpy as np
from sklearn.preprocessing import OneHotEncoder
data.loc[data["Disease"] == "AD", "Disease"] = 0
data.loc[data["Disease"] == "PD", "Disease"] = 1
data.loc[data["Disease"] == "Common", "Disease"] = 2
y=data["Disease"]
enc = OneHotEncoder()
Y = enc.fit_transform(y[:, np.newaxis]).toarray()
