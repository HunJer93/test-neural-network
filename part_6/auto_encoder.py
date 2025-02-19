# Auto Encoders

# import libraries 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# import the datasets for movies, users, and ratings
movies = pd.read_csv('ml-1m/movies.dat', sep= '::', header= None, engine= 'python', encoding= 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep= '::', header= None, engine= 'python', encoding= 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep= '::', header= None, engine= 'python', encoding= 'latin-1')

# prepare the training sets and test sets for analysis
# dataset is delimited by a tab. There are other data sets in the folder for more accurate analysis,
# but for the sake of example, we're only training this model with one training set
training_set = pd.read_csv('ml-100k/u1.base', delimiter= '\t')
training_set = np.array(training_set, dtype= 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter= '\t')
test_set = np.array(test_set, dtype= 'int')

# get the number of users and movies by getting the maximum user id and movie id
user_column_index = 0
movie_column_index = 1
nb_users = int(max(max(training_set[:,user_column_index]), max(test_set[:,user_column_index])))
nb_movies = int(max(max(training_set[:,movie_column_index]), max(test_set[:,movie_column_index])))

# convert the data into an array with users in rows and movies in columns
# used to group the movie ratings for each user by the user id
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        # create array with placeholder 0 values for ratings
        # and add in the id_ratings 
        ratings = np.zeros(nb_movies)
        ratings[id_movies -1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# convert the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# create architecture of the stacked auto encoder
class SAE(nn.Module):
    
    def __init__(self, ):
        super(SAE, self).__init__()
        # create a first hidden using the linear module.
        # created with the number of movies, 20 hidden nodes
        self.fc1 = nn.Linear(nb_movies, 20)
        # second full connection with 10 neurons
        # 20 is used to connect the layers from fc1 (first full connection)
        self.fc2 = nn.Linear(20, 10)
        # fc3 onward starts to decode, so mirror the encoding layers on the first half (nb_movies, 20, 10, 20, nb_movies)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        # using Sigmoid vs. Rectifier, but try with both!
        self.activation = nn.Sigmoid()
    
    # method to run an input through the encoding/decoding layers of the auto encoder
    def forward(self, input_vector):
        input_vector = self.activation(self.fc1(input_vector))
        input_vector = self.activation(self.fc2(input_vector))
        input_vector = self.activation(self.fc3(input_vector))
        input_vector = self.fc4(input_vector)
        return input_vector

# call the auto encoder    
sae = SAE()
# calculate the error with mean square error loss 
criterion = nn.MSELoss()
# create the optimizer with all the values of the auto encoder, the learning rate (try other numbers), and the weight decay (used to regulate the convergence)
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)


# train the stacked auto encoder

# define the number of epochs
nb_epochs = 200

# loop through each epoch
for epoch in range(1, nb_epochs + 1):
    # create the train loss to calculate loss through each epoch
    train_loss = 0
    # count number of users that rated a movie. Created as a float because the root mean squared error (RMSE) is calculated as a float
    s = 0.
    # loop through all of the users that rated a movie
    for id_user in range(nb_users):
        # get the inputs (ratings) of the user from the training set
        # use Variable().unsqueeze(0) to format the data set (array) into a torch tensor (a batch of arrays)
        input = Variable(training_set[id_user]).unsqueeze(0)
        # get the original input before we manipulate the data
        target = input.clone()
        # filter out users that didn't rate any movies for efficiency
        if torch.sum(target.data > 0) > 0:
            # get the output of predicted ratings using the stacked auto encoder, and the user's ratings (input)
            output = sae(input)
            # use .requires_grad = False to apply stochastic gradient descent to the input and not the target (used because this is a shallow clone of the input variable)
            target.requires_grad = False
            # filter out non-zero values from output for optimization
            output[target == 0] = 0
            # calculate the loss error using vector of target vs real ratings (prediction vs. truth)
            loss = criterion(output, target)
            # create mean corrector to calculate the mean of all movies rated and not rated 
            # use 1 ^-10 (1e-10) to make the output a real number and to prevent computations that result in infinity
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            # call backward method for the loss to indicate if we need to increase/decrease the weights
            loss.backward()
            # update the training loss with the difference between the real/predicted rating (calculating the RMSE)
            train_loss += np.sqrt(loss.item()*mean_corrector)
            # increment the number of users that rating a movie
            s += 1.
            # apply the optimizer to update the weights after the RMSE has been applied to the loss
            optimizer.step()
    # output the average training loss for each epoch (train loss divided by the number of computations (users who rated a movie))
    # the difference of 1 in the average loss means that we are off by 1 star out of 5 for our ratings (lower the more accurate we are)
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss/s))
    
# testing the stacked autoencoder

# similar logic to training, but no back propagation (remove back poropagation and optimizer logic)
    test_loss = 0
    s = 0.

for id_user in range(nb_users):
    # get the input from the TRAINING SET so that we can predict what the user might like in the test set.
    input = Variable(training_set[id_user]).unsqueeze(0)
    # this time, use the test set data as the target
    target = Variable(test_set[id_user]).unsqueeze(0)
    # filter out users that didn't rate any movies for efficiency
    if torch.sum(target.data > 0) > 0:
        # use the training set to predict output
        output = sae(input)
        # use .requires_grad = False to apply stochastic gradient descent to the input and not the target (used because this is a shallow clone of the input variable)
        target.requires_grad = False
        # filter out non-zero values from output for optimization
        output[target == 0] = 0
        # calculate the loss error using vector of target vs real ratings (prediction vs. truth)
        loss = criterion(output, target)
        # create mean corrector to calculate the mean of all movies rated and not rated 
        # use 1 ^-10 (1e-10) to make the output a real number and to prevent computations that result in infinity
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        
        # no back propagation because we are not training

        # update the test loss with the difference between the real/predicted rating (calculating the RMSE)
        test_loss += np.sqrt(loss.item()*mean_corrector)
        # increment the number of users that rating a movie
        s += 1.

# output the average test loss for each epoch
# the difference of 1 in the average loss means that we are off by 1 star out of 5 for our ratings (lower the more accurate we are)
print('test loss: ' + str(test_loss/s))
