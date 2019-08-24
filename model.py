import pandas as pd  
import numpy as np  

dataset = pd.read_csv('Foodtruck.csv')  

features = dataset.iloc[:, :-1].values  
labels = dataset.iloc[:, 1].values 

from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(features, labels) 

x=[3.07]
x=np.array(x)
x=x.reshape(1,1)
print(regressor.predict(x))

import pickle

with open('zomato_profit.pkl', 'wb') as file:
    pickle.dump(regressor, file)
