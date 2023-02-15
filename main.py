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

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,Y,test_size=0.2)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras import regularizers
from keras.optimizers import RMSprop
model = Sequential()
model.add(Dense(64, input_dim = 1437, activation = 'relu'))
model.add(LeakyReLU(alpha=0.05))
model.add(Dense(128))
model.add(LeakyReLU(alpha=0.09))
model.add(Dense(256, activation = 'relu',kernel_regularizer=regularizers.l2(0.09)))
model.add(Dropout(0.3))
model.add(Dense(128))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'RMSprop', metrics = ['accuracy'])

model.summary()

model.fit(x_train,y_train,epochs=1000,verbose=1)

loss,accuracy=model.evaluate(x_test,y_test)


