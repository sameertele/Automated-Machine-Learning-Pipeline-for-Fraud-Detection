#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Scaling the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    data = pd.read_csv(os.path.join('data', 'raw', 'creditcard.csv'))
    
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    processed_data_dir = os.path.join('data', 'processed')
    os.makedirs(processed_data_dir, exist_ok=True)
    pd.DataFrame(X_train).to_csv(os.path.join(processed_data_dir, 'X_train.csv'), index=False)
    pd.DataFrame(X_test).to_csv(os.path.join(processed_data_dir, 'X_test.csv'), index=False)
    pd.DataFrame(y_train).to_csv(os.path.join(processed_data_dir, 'y_train.csv'), index=False)
    pd.DataFrame(y_test).to_csv(os.path.join(processed_data_dir, 'y_test.csv'), index=False)

