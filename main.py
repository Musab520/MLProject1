import pandas as pd
import numpy as np
import seaborn as sns

#Load the data
df = pd.read_csv('Customer Churn.csv')

#View the data
pd.set_option('display.max_columns', None)
print(df.head())
print(df.info())
print(df.describe())