#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from azureml.core import Workspace, Dataset, Experiment
from azureml.train.automl import AutoMLConfig

def train_model(X_train, y_train):
    ws = Workspace.from_config()

    experiment = Experiment(ws, 'fraud-detection-experiment')

    automl_config = AutoMLConfig(
        task='classification',
        training_data=X_train,
        label_column_name='Class',
        primary_metric='AUC_weighted',
        iterations=10,
        n_cross_validations=5,
        enable_early_stopping=True
    )

    run = experiment.submit(automl_config, show_output=True)
    best_run, fitted_model = run.get_output()

    return best_run, fitted_model

if __name__ == "__main__":
    X_train = pd.read_csv(os.path.join('data', 'processed', 'X_train.csv'))
    y_train = pd.read_csv(os.path.join('data', 'processed', 'y_train.csv'))

    best_run, fitted_model = train_model(X_train, y_train)
    print(f"Best model: {fitted_model}")

