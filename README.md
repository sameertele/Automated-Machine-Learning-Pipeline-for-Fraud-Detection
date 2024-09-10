# **Automated Machine Learning Pipeline for Fraud Detection**

This project demonstrates how to set up an automated machine learning (ML) pipeline for fraud detection using Azure Machine Learning (AML). The pipeline consists of multiple stages including data ingestion, preprocessing, model training, and deployment. The project is built using Python and Azure ML SDK and is designed to automate retraining and deployment with new data.

## **Dataset**

For this project, we use the **[Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)** from Kaggle. The dataset contains transactions made by European cardholders, with 492 fraudulent transactions out of 284,807 total transactions.

### **Key Features**
- **Time**: Number of seconds between this transaction and the first transaction in the dataset.
- **V1-V28**: Principal components from PCA.
- **Amount**: The transaction amount.
- **Class**: 0 = No Fraud, 1 = Fraud.
