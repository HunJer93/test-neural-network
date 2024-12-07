# Recurrent Neural Network

# import libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# part 1: data preprocessing
# import the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
# get the data from the 2nd column (Open column) in an array (: is the range, and 1:2 will select the first column)
training_set = dataset_train.iloc[:, 1:2].values

# choosing between standardisation feature scaling vs normalization feature scaling
# standardisation = x - mean(x) / standard deviation(x)
# normalization = x - min(x) / max(x) - min(x)
# since we are using a signmoid activation function in the output layer, we are going to use normalization

# Feature Scaling

# using normalization feature scaling
sc = MinMaxScaler(feature_range= (0, 1))

# apply normalization to scale data
training_set_scaled = sc.fit_transform(training_set)

# Create a data structure with 60 timesteps and 1 output
# looks at 60 preivous iterations to make predictions
# timesteps will need to be fiddled with to get the model fitted right (prevent overfitting or underfitting)

# 60 previous stock prices
x_train = []
# prediction of next stock price
y_train = []

# iterate starting on 60th iteration and going to end of training set   
for i in range(60, len(training_set_scaled)):
    x_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

# format into arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# reshaping dimensions to include batch size, time steps (iterations in step above of 60), and number of indicators
# documentation in "sequences" in https://keras.io/api/layers/recurrent_layers/rnn/
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Part 2: build the RNN


# Part 3: making the pedictions and visualizing the results with matplot