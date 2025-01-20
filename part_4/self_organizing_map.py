# Self Organizing Map

# purpose of script is to go through a list of users that applied for a credit card
# and find the outliers that might be potential fraud (especially the ones that were approved for the card)

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
# used this import to solve error with Qt platform plugin not initialized
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd

# import the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
# get variables in the data set (except for the last column for approvals and customer id (id irrelivant))
input_params = dataset.iloc[:, :-1].values
# get the last variable checkig if users were approved (what we're testing for).
output_param = dataset.iloc[:, -1].values


# Feature Scaling
# use feature scaling to normalize the data set between 0 and 1 values
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range= (0, 1))
# fit the normalization to input values
sc.fit_transform(input_params)

# Train the SOM
# using MiniSom for map
# for anaconda use: 'conda install conda-forge::minisom'
from minisom import MiniSom

# train the self organizing map on 
# use a 10x10 grid for our smaller data set
# the input_len is the number of all input and outputs in our dataset
# learning rate set to default (can beef up to increase accuracy)
som = MiniSom(x = 10, y = 10, input_len= (len(dataset.columns) - 1), learning_rate= 0.5)
# set random starting weights
som.random_weights_init(input_params)

# train the self organizing map on the inputs (2nd param is number of iterations)
som.train_random(input_params, num_iteration= 100)



# Plot the Self Organizing Map Results
from pylab import bone, pcolor, colorbar, plot, show
bone()
# get the distance map from the self organizing map, and use .T to transpose to 2D grid
pcolor(som.distance_map().T)
colorbar()

plt.figure(figsize=(10,10))
plt.pcolor(som.distance_map().T, cmap='gist_yarg')
plt.colorbar()
plt.show()