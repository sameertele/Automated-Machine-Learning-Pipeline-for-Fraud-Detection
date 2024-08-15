#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd

def load_data(data_path):
    data = pd.read_csv(data_path)
    return data

if __name__ == "__main__":
    data_path = os.path.join('data', 'raw', 'creditcard.csv')
    data = load_data(data_path)
    print(f"Data shape: {data.shape}")
    print(data.head())

