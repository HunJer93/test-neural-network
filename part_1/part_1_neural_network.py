import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score


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
# use fit_transform to set scaling, and then ONLY transform after that
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
# using sigmoid to predict binary output. We would use soft max for multiple variables output
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

###### END OF A.N.N ######

###### START OF TRAINING NETWORK ######
# compile the network for training
# using stochastic gradient descent to update weights (adam)
# since we want binary output, we need 'binary_crossentropy'. For non-binary use crossentropy
# only tracking the accuracy metric, but we could add others to the array if we wanted more metrics
ann.compile(optimizer= 'adam', loss='binary_crossentropy' , metrics= ['accuracy'])

# train the ann on the training set
# using batch size default of 32, and using batch training because it more accurately trains an ann but updating the weights in batches
# epochs are the number of times the training runs. The higher the number, the greater the accuracy.
ann.fit(x_train, y_train, batch_size = 32, epochs = 100)

###### END OF TRAINING NETWORK ######

###### START OF USING NETWORK FOR PREDICTIONS ######

# use the trained model to make predictions (always use double brackets with predict method)
# predict a user will stay or leave. Use transform when adding new values, and only use fit_transform for first value.
single_prediction = ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2 , 1, 1, 50000]]))

print("Will the customer leave the bank? (predicted) ", single_prediction > 0.5) # use 50/50 for leave (True) or stay (False)

# predict the test results from our test data set
y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)
test_prediction = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
print("test prediction vs actual prediction: ", test_prediction)

###### END OF USING NETWORK FOR PREDICTIONS ######

###### START OF CONFUSION MATRIX ######
# create confusion matrix to give percentage accuracy from neural network to actual results(test data)
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix: ", cm)
print("accuracy score: ", accuracy_score(y_test, y_pred))
###### END OF CONFUSION MATRIX ######