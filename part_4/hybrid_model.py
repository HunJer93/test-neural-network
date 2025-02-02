# Make a hybrid deep learning model 
# the logic is currently broken due to conflicts with tensorflow and numpy causing np.object error with Numpy scalar

# import libraries
import pdb
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

# deep learning libraries
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler

class HybridModel: 
    
    def __init__(self, file_name):
        self.file_name = file_name
        self.dataset = None
        self.input_params = []
        self.deep_learning_input_params = []
        self.output_params = []
        self.self_org_map = None
        self.scaler = None
        self.fraud_list = None
        
    
    # creates a trained self organizing map for the file type
    def create_som(self):
            
        # import dataset from CSV
        self.dataset = pd.read_csv(self.file_name)

        # get input variables for som and deep learning model
        self.input_params = self.dataset.iloc[:, :-1].values
        self.deep_learning_input_params = self.dataset.iloc[:, 1:].values

        # get test value
        self.output_params = self.dataset.iloc[:, -1].values

        # set feature scaling normalized between 0 and 1
        self.scaler = MinMaxScaler(feature_range= (0, 1))
        self.scaler.fit_transform(self.input_params)

        # train the self organizing map
        self_org_map = MiniSom(x = 10, y = 10, input_len= (len(self.dataset.columns) - 1), sigma= 1.0, learning_rate= 0.5)
        self_org_map.random_weights_init(self.input_params)
        self_org_map.train_random(self.input_params, num_iteration= 100)
        self.self_org_map = self_org_map
        print("Self Organizing Map traning complete for dataset " + self.file_name)
        
    def create_deep_learning_model(self):
        # independent variable for model
        # using csv data minus output (dependent variable) and user id
        customers = self.deep_learning_input_params
        
        # create dependent by looping through dataset and matching customer id to id in fraud list
        # replace the 0 with 1 for true
        is_fraud = np.zeros(len(self.dataset))
        for i in range(len(self.dataset)):
            if self.dataset.iloc[i, 0] in self.fraud_list:
                is_fraud[i] = 1
        print(is_fraud)
        
        sc = StandardScaler()
        customers = sc.fit_transform(customers)
        
        # initialize ANN
        classifier = Sequential()
        
        # add the input layer and first hidden layer
        classifier.add(Dense(units = 2, kernel_initializer= 'uniform', activation= 'relu', input_dim = 15))
        # add the output layer
        classifier.add(Dense(units = 1, kernel_initializer= 'uniform', activation= 'sigmoid'))
        # compile ann
        classifier.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])
        # fit ANN to training set
        classifier.fit(customers, is_fraud, batch_size= 1, epochs= 2)
        
        outcome_prediction = classifier.predict(customers)
        print(outcome_prediction)
        
        
        
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
        self.calculate_frauds(sanitized_values)

    
    # use SOM to find frauds in the dataset
    def calculate_frauds(self, som_coordinates):
        # find the frauds in the list
        mappings = self.self_org_map.win_map(self.input_params)
        # pull frauds using mean inter-neuron distance
        # get coordinates from input map
        frauds = []
        # left off here! Need to concatenate frauds list at coordinates on the map given
        frauds = np.concatenate((mappings[(som_coordinates[0][0], som_coordinates[0][1])],mappings[(som_coordinates[1][0], som_coordinates[1][1])]), axis= 0)   
        self.fraud_list = self.scaler.inverse_transform(frauds)
        print(self.fraud_list)
            
        
        
    def clean_input(self, input):
        unsanitized_values = input.split(':')
        unsanitized_values = list(map(lambda value: value.strip(), unsanitized_values))
        # created 2D array with coordinates
        cleaned_values = []
        for coordinates in unsanitized_values:
            values = coordinates.split(',')
            values = map(lambda coordinate: int(coordinate), values)
            cleaned_values.append(list(values))
            
        return cleaned_values        
        



# Part 1: Identify the Frauds with a Self Organizing Map
user_input = input('Enter the name of the csv file you want to run analysis on: ')

model = HybridModel(user_input)
model.create_som()
model.print_map()
print('Enter the locations in coordinates from the map (light areas) to determine the frauds')
print('Separate coordinates with a comma and users with a colon : (2 values needed)')
user_input = input("Enter values here: ")
model.find_frauds(user_input)


# Part 2: Use Self Organizing Map to predict frauds using Supervised Deep Learning
model.create_deep_learning_model()
        
        

        
    
        







