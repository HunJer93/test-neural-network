import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


###### START DATA PREPROCESSING ######

# import the dataset from CSV
dataset = pd.read_csv('Churn_Modelling.csv')
# create matrix, ignoring irrelevant columns (CustomerId and Surname)
x = dataset.iloc[:, 3:-1].values # retrieves columns CreditScore onward, excludes Existed (what we are testing for)
y = dataset.iloc[:, -1].values # retrieves Exited only (what we are testing for)


# encode categorical data (Gender, Geography, etc.)

# label encoding the "Gender" column to 0 and 1 from "Female" and "Male"
le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])

# hot encode 'Geography' column
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder="passthrough")
x = np.array(ct.fit_transform(x))

# split dataset into Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 0)

# feature scaling 
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

###### END DATA PREPROCESSING ######

###### START OF A.N.N ######
# instantiate a shallow neural network
ann = tf.keras.models.Sequential()

# create input layer and first hidden layer
# When adding nodes to the network there is no rule of thumb, so try adding different #s of units going forward
# using rectifier activation function with 'relu'
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# create second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# create output layer
# using 1 unit for exited output. If we wanted more variables output, we would increase the units.
# using sigmoid to predict binary output
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

###### END OF A.N.N ######