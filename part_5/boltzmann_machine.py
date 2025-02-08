# Restricted Boltzmann Machine for recommending if a user will like a movie or not
# data set for movies come from https://grouplens.org/datasets/movielens/

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
# torch FloatTensor only accepts a list of lists
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# convert the ratings into binary ratings (1 liked and 0 disliked)
# replace existing 0 ratings with -1 
training_set[training_set == 0] = -1
training_set[training_set == 1 ] = 0
training_set[training_set == 2 ] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1 ] = 0
test_set[test_set == 2 ] = 0
test_set[test_set >= 3] = 1


# create the architecture of the neural network
class RBM():
    
    # nv is visible nodes and nh is the hidden nodes
    def __init__(self, nv, nh):
        self.weights = torch.randn(nh, nv) # initalized the random weights for nodes
        self.bias_hidden = torch.randn(1, nh) # added 1 to bias to created a 2D tensor
        self.bias_visible = torch.randn(1, nv) # added 1 to bias to created a 2D tensor
    
    # method used to sample the hidden nodes in the network
    # and figure out the probability that the hidden node is activated
    # based on what we can see in the visible node
    def sample_h(self, visible_neurons):
        # multiply the visible neurons by the current weights (converted to tensors with .t())
        weight_visible = torch.mm(visible_neurons, self.weights.t())
        # calculate the activation threshold for like or dislike
        activation = weight_visible + self.bias_hidden.expand_as(weight_visible)
        # use a sigmoid function to create binary output
        probability_of_hidden_activation = torch.sigmoid(activation)
        
        # return the probability and sampling of probability in our data set using bernoulli's principle for sampling
        return probability_of_hidden_activation, torch.bernoulli(probability_of_hidden_activation)
        

