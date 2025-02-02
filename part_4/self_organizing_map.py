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
som = MiniSom(x = 10, y = 10, input_len= (len(dataset.columns) - 1), sigma= 1.0, learning_rate= 0.5)
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

# add markers to the grid to show which customers are the outliers
# takes the winning neuron of the row and plots in the center (0.5 added to center the plotted value), and uses the index to link approved/denied value from the output_params
markers = ['o', 's'] # if fraud, see a red circle; otherwise green square
colors = ['r', 'g']
for index, customer in enumerate(input_params):
    winning_node = som.winner(customer)
    plot(winning_node[0] + 0.5,
         winning_node[1] + 0.5,
         markers[output_param[index]],
         markeredgecolor= colors[output_param[index]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# find the frauds in the list
mappings = som.win_map(input_params)
# frauds need to be pulled from the values that have the Mean Inter-Neuron Distance (MID) at 1 (the white values on the output map)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,9)]), axis=0)
frauds = sc.inverse_transform(frauds)


print(frauds)

