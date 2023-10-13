# Import required modules
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np


# Loading the dataset 
Insurance_cost_dataset = pd.read_csv("insurance.csv")
# Show the dataset
Insurance_cost_dataset.head()
Insurance_cost_dataset.tail()
# Show the dataset shape
Insurance_cost_dataset.shape
# Show some statistical info about the dataset
Insurance_cost_dataset.describe()


# Check about the none(missing) values in the dataset to decide to make a data cleaning or not
Insurance_cost_dataset.isnull().sum()


# Show and plot the number of values in sex, smoker and region and its repetitions
plt.figure(figsize=(5,10))
Insurance_cost_dataset['sex'].value_counts()
sns.catplot(x = 'sex',data = Insurance_cost_dataset,kind='count')
plt.show()
Insurance_cost_dataset['children'].value_counts()
sns.catplot(x = 'children',data = Insurance_cost_dataset,kind='count')
plt.show()
Insurance_cost_dataset['smoker'].value_counts()
sns.catplot(x = 'smoker',data = Insurance_cost_dataset,kind='count')
plt.show()
Insurance_cost_dataset['region'].value_counts()
sns.catplot(x = 'region',data = Insurance_cost_dataset,kind='count')
plt.show()

# Plot the distribution of Age column
plt.figure(figsize=(5,5))
sns.distplot(Insurance_cost_dataset['bmi'],color="red")
# Plot the distribution of Age column
plt.figure(figsize=(5,5))
sns.distplot(Insurance_cost_dataset['age'],color="green")
# Plot the distribution of charges(output) 
plt.figure(figsize=(5,5))
sns.distplot(Insurance_cost_dataset['charges'],color="blue")

# Find the correlation betwee various features in the dataset
correlation_values = Insurance_cost_dataset.corr()
# Plot the correlation of the dataset
plt.figure(figsize=(10,10))
sns.heatmap(correlation_values,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Blues')



# create a label Encoder
le = LabelEncoder()
# Label each textual columns into numeric column
datatypes = Insurance_cost_dataset.dtypes
print(datatypes)
for IndexDtype in range(len(datatypes)):
    if datatypes[IndexDtype]=="object":
        Insurance_cost_dataset.iloc[:,IndexDtype] = le.fit_transform(Insurance_cost_dataset.iloc[:,IndexDtype]) 



# Split the data into input and label data
X = Insurance_cost_dataset.drop(columns=['charges'],axis=1)
Y = Insurance_cost_dataset['charges']
print(X)
print(Y)
# Split the data into train and test data
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.8,random_state=6)
print(X.shape,x_train.shape,x_test.shape)
print(Y.shape,y_train.shape,y_test.shape)



# Create the Linear Regression model and train it
LRModel = LinearRegression()
LRModel.fit(x_train,y_train)
# Make the model predict on train data
predicted_train_values = LRModel.predict(x_train)
predicted_test_values = LRModel.predict(x_test)
# Avaluate the model
accuracy_train_data = r2_score(predicted_train_values,y_train)
accuracy_test_data = r2_score(predicted_test_values,y_test)
# Show the accuracy on the train and test data prediction
print(accuracy_train_data,accuracy_test_data)



# Build a predictive system
input_data = (46,0,33.44,1,0,2)
# Convert input data into 1D numpy array
input_array = np.array(input_data)
# Convert 1D input array into 2D
input_2D_array = input_array.reshape(1,-1)
# Predict input value
print(LRModel.predict(input_2D_array)[0])

