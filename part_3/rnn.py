# Recurrent Neural Network

# import libraries 
import numpy as np
import matplotlib.pyplot as plt
# used to resolve this error: This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.
import matplotlib
matplotlib.use('TKAgg') # used different graphing library to resolve above error ^
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

######################
# part 1: data preprocessing
######################

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
previous_step_count = 60

# 60 previous stock prices
x_train = []
# prediction of next stock price
y_train = []

# iterate starting on 60th iteration and going to end of training set   
for i in range(previous_step_count, len(training_set_scaled)):
    x_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

# format into arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# reshaping dimensions to include batch size, time steps (iterations in step above of 60), and number of indicators (1 since we only have a single stock price. We would have more indicators if we had other variables [i.e: Samsung does business with Google. If we had Samsung's stock that would be a 2nd indicator on Google's stock price])
# documentation in "sequences" in https://keras.io/api/layers/recurrent_layers/rnn/
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

###########################
# Part 2: build the RNN
###########################
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

######################
# Part 3: making the pedictions and visualizing the results with matplot
######################

# get the real stock price from test set
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_train.iloc[:, 1:2].values

# get predicted stock price

# concatenate the training and test set stock price ('Open' is the stock price) to get a new dataframe with both columns
# using vertical concatenation (axis = 0) vs horizontal concatination (axis = 1) to get the column data instead of the row. 
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)

# get the upper and lower bounds of the test set.
# since we are using 60 previous iterations for prediction, we need to offset the data set by 60 days in the beginning to be consistent with the test
# use .values to get a numpy array
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values

# reshape format for Numpy 
inputs = inputs.reshape(-1,1)

# scale inputs. We ONLY scale inputs and keep test values the same
inputs = sc.transform(inputs)

# iterate starting on 60th iteration and going to end of training set 
# upper bound set to 80 (60 previous iterations plus the 20 values in the test set)
x_test = []
upper_bound_test = previous_step_count + len(dataset_test)

for i in range(previous_step_count, upper_bound_test):
    x_test.append(inputs[i-60:i, 0])

# format into array
x_test = np.array(x_test)

# reshape to 3D format (similar to train)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# use regressor to predict the results
predicted_stock_price = regressor.predict(x_test)

# inverse the scaling of the regressor
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


############################
# visualize the results
############################

# create plot
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
