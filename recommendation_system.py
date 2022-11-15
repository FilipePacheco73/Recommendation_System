# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 09:16:58 2022

@author: Filipe Pacheco

Code to create a Recommendation System such those ones in Social Network

Based on the post on: https://medium.com/@mervetorkan/association-rules-with-python-9158974e761a

"""

# Libraries - Imports

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Dataset can be downloaded from here https://www.kaggle.com/shazadudwadia/supermarket
df = pd.read_csv("GroceryStoreDataSet.csv", names = ['products'], sep=',')

print(df.head()) # see the first 5 instances
print(df.shape) # check the dimensions of the data

# Split the lines into columns
data = list(df['products'].apply(lambda x:x.split(",")))

# Apriori Algorithm and One-Hot Enconding - Create an easy way to map each instance and its ocurrence
aux = TransactionEncoder() # create object
aux_data = aux.fit(data).transform(data) # transform data into One-Hot Encoding

df = pd.DataFrame(aux_data,columns=aux.columns_) # recreate the original dataframe with a new structure to be applied by the Apriori Algorithm
df = df.replace(False,0)
df = df.replace(True,1)

# Applying Apriori Algorithm
df_apriori = apriori(df, min_support = 0.2, use_colnames = True, verbose = True)
print(df_apriori)

# Checking values using Association Rule function

df_ar = association_rules(df_apriori, metric = 'confidence', min_threshold=0.6)
print(df_ar)