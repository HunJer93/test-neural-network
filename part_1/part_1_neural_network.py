import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# data preprocessing

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

print(x)

