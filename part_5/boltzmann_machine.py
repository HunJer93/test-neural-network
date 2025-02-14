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
 
    # create inverse sampling method for visible nodes
    def sample_v(self, hidden_neurons):
        # multiply the visible neurons by the current weights (converted to tensors with .t())
        weight_hidden = torch.mm(hidden_neurons, self.weights)
        # calculate the activation threshold for like or dislike
        activation = weight_hidden + self.bias_visible.expand_as(weight_hidden)
        # use a sigmoid function to create binary output
        probability_of_visible_activation = torch.sigmoid(activation)
        
        # return the probability and sampling of probability in our data set using bernoulli's principle for sampling
        return probability_of_visible_activation, torch.bernoulli(probability_of_visible_activation)
    
    # v0 = input vector
    # vk = visible nodes after k sampling
    # ph0 = vector probability that hidden nodes are activated (node = 1) given the input vector
    # phk = probability of hidden nodes after k sampling given values of visible node sampling (vk)
    def train(self, v0, vk, ph0, phk):
        # update the weights given sampling
        # weight = (visible vector multiplied by hidden vector activation probability) - (visible sampling vectors multiplied by hidden sampling activation probability) 
        self.weights += torch.mm(v0.t(), ph0).t() - torch.mm(vk.t(), phk).t()
        
        # update the visible bias
        # v_bias = visible vector minus by the sampling
        self.bias_visible += torch.sum((v0 - vk), 0)
        
        # update the visible bias
        # h_bias = hidden vector activation probability minus probability hidden after k sampling
        self.bias_hidden += torch.sum((ph0 - phk), 0)
        
# get the number of visible nodes, hidden nodes, and batch size (can play around with the hidden node and batch size)
nodes_visible = len(training_set[0])
nodes_hidden = 100
batch_size = 100

print('visible: ' + str(nodes_visible) + ' hidden: ' + str(nodes_hidden))
rbm = RBM(nodes_visible, nodes_hidden)

# training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    # calculate the loss in each epoch
    train_loss = 0
    s = 0.
    # iterate through users in batches
    for id_user in range(0, nb_users - batch_size, batch_size):
        # vk = sampling batch (we will update this as we go)
        # start at the user id and get the batch size from the training set (returns 100 users)
        vk = training_set[id_user:id_user + batch_size]
        v0 = training_set[id_user:id_user + batch_size]
        # create probability of hidden nodes from the visible nodes sampled
        # used to create the 'random walk' (Monte Carlo Markov Chain (MCMC))
        ph0,_ = rbm.sample_h(v0)
        
        # use Gibbs sampling (k random walk)
        for k in range(10):
            # sample back and forth between visible and hidden nodes, updating the sampled probability through each iteration
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            # freeze the -1 ratings so that they aren't updated in Gibbs sampling
            vk[v0 < 0] = v0[v0 < 0]
        # get the probability of hidden activation from sampling
        phk,_ = rbm.sample_h(vk)
        # run the training with values created
        rbm.train(v0, vk, ph0, phk)
        
        # calculate the loss by getting the average distance between the first vector and sampled vector for relevant ratings (excluding -1 values that indicate no rating given)
        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
        # increment the counter
        s += 1.
    print('epoch: ' + str(epoch)+ ' loss: ' + str(train_loss/s))
    
    
    
    
# test the RBM


test_loss = 0
s = 0.
# run the model on each of the users in the test set
for id_user in range(nb_users):
    # use the training set inputs to activate the hidden neruons in the network
    # this is then used to predict the outcome of the target 'test set'
    visible_nodes = training_set[id_user:id_user+1]
    input_target = test_set[id_user:id_user+1]
    # no for loop because we only walk to iterate through the test set once
    # only walk through the values in the test set once due to MCMC training that occured in training (this is called the 'blind walk')
    # 'we were trained to walk 100 steps blindfolded, so it is more accurate to walk one step blindfolded to prevent diviation"
    
    # using this if block to filter out the -1 values that represented no user feedback
    if len(input_target[input_target>=0]) > 0:
        # use Gibbs sampling to get a hidden node and a visible node
        _,hidden_nodes = rbm.sample_h(visible_nodes)
        _,visible_nodes = rbm.sample_v(hidden_nodes)
        test_loss += torch.mean(torch.abs(input_target[input_target>=0] - visible_nodes[input_target>=0]))
        s += 1. 
# get the average loss of the loss divided by the number of iterations
# the loss determines how often the model is wrong so .25 outcome is correct at predicting 3 out of 4 times. 
print('test loss: '+str(test_loss/s))