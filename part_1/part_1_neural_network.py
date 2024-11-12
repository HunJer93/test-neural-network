import numpy as np
import pandas as pd
import tensorflow as tf

print(tf.__version__)

# data preprocessing

# import the dataset from CSV
dataset = pd.read_csv('Churn_Modelling.csv')
# create matrix 
x = dataset.iloc