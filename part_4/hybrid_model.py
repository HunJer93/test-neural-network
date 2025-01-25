# Make a hybrid deep learning model 

# import libraries
import __main__
import numpy as np
import matplotlib.pyplot as plt
# used this import to solve error with Qt platform plugin not initialized
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
# for feature scaling 
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
# used to display output from self org map
from pylab import bone, pcolor, colorbar, plot, show

class HybridModel: 
    
    def __init__(self, file_name):
        self.file_name = file_name
        self.input_params = []
        self.output_params = []
        self.self_org_map = None
        
    
    # creates a trained self organizing map for the file type
    def create_som(self):
            
        # import dataset from CSV
        dataset = pd.read_csv(self.file_name)

        # get input variables
        self.input_params = dataset.iloc[:, :-1].values

        # get test value
        self.output_params = dataset.iloc[:, -1].values

        # set feature scaling normalized between 0 and 1
        scaler = MinMaxScaler(feature_range= (0, 1))
        scaler.fit_transform(self.input_params)

        # train the self organizing map
        self_org_map = MiniSom(x = 10, y = 10, input_len= (len(dataset.columns) - 1), sigma= 1.0, learning_rate= 0.5)
        self_org_map.random_weights_init(self.input_params)
        self_org_map.train_random(self.input_params, num_iteration= 100)
        self.self_org_map = self_org_map
        print("Self Organizing Map traning complete for dataset " + self.file_name)
        
    def print_map(self):
        som = self.self_org_map
        output_params = self.output_params
        input_params = self.input_params
        
        bone()
        # display distance map in 2D
        pcolor(self.self_org_map.distance_map().T)
        colorbar()
        # add markers to the grid to show which customers are the outliers
        # takes the winning neuron of the row and plots in the center (0.5 added to center the plotted value), and uses the index to link approved/denied value from the output_params
        markers = ['o', 's'] # if fraud, see a red circle; otherwise green square
        colors = ['r', 'g']
        for index, customer in enumerate(input_params):
            winning_node = som.winner(customer)
            plot(winning_node[0] + 0.5,
                winning_node[1] + 0.5,
                markers[output_params[index]],
                markeredgecolor= colors[output_params[index]],
                markerfacecolor = 'None',
                markersize = 10,
                markeredgewidth = 2)
        show()
    
        
    def find_frauds(self, user_input):
        
        # clean input data
        sanitized_values = self.clean_input(user_input)
        
        # calculate frauds
        # self.calculate_frauds()

    
    # use SOM to find frauds in the dataset
    def calculate_frauds(self):
        # create SOM if not created yet
        if self.self_org_map is None:
            self.create_som()
        
        # find the frauds in the list
        mappings = self.self_org_map.win_map(self.input_params)
        # pull frauds using mean inter-neuron distance
        
        # left off here! 
        # need to work out how to get values out of string for mapping
        # frauds = np.concatenate((mappings[]))
        
    def clean_input(self, input):
        unsanitized_values = input.split(':')
        unsanitized_values = list(map(lambda value: value.strip(), unsanitized_values))
        print(unsanitized_values)
        


# Part 1: Identify the Frauds with a Self Organizing Map
user_input = input('Enter the name of the csv file you want to run analysis on: ')

model = HybridModel(user_input)
model.create_som()
model.print_map()
print('Enter the locations in coordinates from the map (light areas) to determine the frauds')
print('Group the users in parentheses () and separate with a colon : if more than one')
user_input = input("Enter values here: ")
model.find_frauds(user_input)


# Part 2: Use Self Organizing Map to predict frauds using Supervised Deep Learning
        
        

        
    
        







