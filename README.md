# Medical-Insurance-Cost-prediction
This project focuses on predicting medical insurance costs using machine learning techniques, specifically linear regression. 
It is essential for individuals and insurance companies to have an estimate of the medical insurance cost, which can help in planning and budgeting for healthcare expenses.

## Prerequisites
Before you run this code, make sure you have the following prerequisites in place:
-Python 3.x installed on your system.
-Required Python libraries: pandas, scikit-learn, matplotlib, seaborn, and numpy. You can install these libraries using pip: pip install pandas scikit-learn matplotlib seaborn numpy
-The dataset file insurance.csv included in the same directory as the code. This dataset contains information such as age, gender, BMI, number of children, smoker status, region, and insurance charges.

## Overview
The project consists of the following steps:
- Data Loading and Exploration: The code starts by importing necessary libraries and loading the insurance dataset. It then displays the first and last few rows of the dataset, its shape, and some basic statistical information.
- Data Cleaning: It checks for missing values in the dataset to decide if data cleaning is necessary.
- Data Visualization: The code provides visualizations of the distribution of values for columns 'sex', 'children', 'smoker', 'region', 'bmi', 'age', and 'charges' to better understand the data.
- Correlation Analysis: It calculates the correlation between various features in the dataset and visualizes it using a heatmap.
- Data Preprocessing: Textual columns are converted into numeric using Label Encoding.
- Data Splitting: The dataset is split into input features (X) and the target variable (Y). It is then further split into training and testing data.
- Model Building and Training: A Linear Regression model is created and trained on the training data.
- Model Evaluation: The model's performance is evaluated on both the training and testing data, and the R-squared score is computed. The accuracy of the model on the training and testing data is 70%.
- Prediction: The code includes a predictive system. Given an input array with features like age, gender, BMI, etc., it predicts the insurance cost using the trained model.

## Contribution
Contributions to this project are welcome! If you have ideas for improving the accuracy of the model, adding new features, or enhancing the visualizations, feel free to contribute.
You can create issues, fork the repository, and submit pull requests to make this project even better.
