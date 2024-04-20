# Breast-Cancer-Prediction-Using-ML

This repository contains a Python script for predicting breast cancer diagnosis using various machine learning classifiers. The script utilizes the scikit-learn library to implement classifiers such as K-Nearest Neighbors (KNN), Decision Tree, Support Vector Machine (SVM), Naive Bayes, and Random Forest. The goal is to build and evaluate models that can accurately classify breast tumors as malignant or benign based on input features.

## Overview

In this project, we aim to predict breast cancer diagnosis using a dataset containing various features extracted from digitized images of breast mass. The dataset includes features such as mean radius, mean texture, mean area, and more, which are used to train and evaluate the machine learning models. Key steps include:

- Data Loading: The script loads the breast cancer dataset from a CSV file using pandas, dropping irrelevant columns like ID and encoding the target variable (diagnosis) as numeric labels.
- Data Preprocessing: The dataset is preprocessed to handle missing values, split into feature (X) and target (Y) sets, and further divided into training and testing sets.
- Model Training: Several classifiers including KNN, Decision Tree, SVM, Naive Bayes, and Random Forest are trained on the training data using scikit-learn's API.
- Model Evaluation: The trained models are evaluated on the testing data using metrics such as F1-score to assess their performance in predicting breast cancer diagnosis.
- ROC Curve Visualization: Receiver Operating Characteristic (ROC) curves are plotted for each classifier to visualize their performance in terms of true positive rate (sensitivity) and false positive rate (1-specificity).

## Dependencies

- Python 3
- pandas
- numpy
- matplotlib
- scikit-learn
- plotnine
