# Recurrent Neural Network

# import libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# keras libraries for neural network
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# numpy version had to be downgraded to 1.18.5 in order to work with Keras
# otherwise was throwing the following error:
# NotImplementedError: Cannot convert a symbolic Tensor (sequential/lstm/strided_slice:0) to a numpy array. This error may indicate that you're trying to pass a Tensor to a NumPy call, which is not supported
print(np.__version__)

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

# reshaping dimensions to include batch size, time steps (iterations in step above of 60), and number of indicators (1 since we only have a single stock price. We would have more indicators if we had other variables [i.e: Samsung does business with Google. If we had Samsung's stock that would be a 2nd indicator on Google's stock price])
# documentation in "sequences" in https://keras.io/api/layers/recurrent_layers/rnn/
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Part 2: build the RNN
# use Keras libraries

# initialize the RNN
regressor = Sequential()

# create first LSTM layer
# LSTM accepts number of units (number of LSTM cells), return sequence (true or false [true when adding new layers to return values]), and input shape (number of iterations and number of indicators)
regressor.add(LSTM(units = 50, return_sequences= True, input_shape = (x_train.shape[1], 1)))

# add dropout to prevent over fitting to our data set
# standard practice is to drop 20% of the nodes in the LSTM layer to prevent overfitting
regressor.add(Dropout(0.2))


# repeat by adding a new layer with dropout (no input needed)
regressor.add(LSTM(units = 50, return_sequences= True))
regressor.add(Dropout(0.2))

# 3rd layer
regressor.add(LSTM(units = 50, return_sequences= True))
regressor.add(Dropout(0.2))

# 4th layer with no return sequence because it is the last layer
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# output layer. Outputs 1 dimension (stock price) [would be more if there were multiple variables]
regressor.add(Dense(units= 1))

# compile the RNN with optimizer (usually RMSprop, but can also use Adam) and loss function (mean squared error)
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# fit the RNN to the training set
# fit requires inputs of the training set, expected output of the training set, number of epochs (play around with the number), and batch size
regressor.fit(x_train, y_train, epochs = 100, batch_size = 32)


# Part 3: making the pedictions and visualizing the results with matplot