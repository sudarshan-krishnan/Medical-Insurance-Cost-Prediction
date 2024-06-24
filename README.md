# Insurance Cost Prediction Using Linear Regression

This project demonstrates how to predict insurance costs using a linear regression model. By leveraging Python and its powerful data science libraries, we import and analyze the data, preprocess it, and then build and evaluate a predictive model.

## Key Features
- **Data Collection & Analysis:** Load and explore the insurance dataset.
- **Data Visualization:** Use plots to understand data distribution and relationships.
- **Preprocessing:** Encode categorical variables and split data into features and target variables.
- **Model Training:** Train a linear regression model to predict insurance costs.
- **Model Evaluation:** Evaluate the model's performance using metrics like R-squared.

## Components
- Python libraries: NumPy, pandas, matplotlib, seaborn, scikit-learn.
- Jupyter Notebook for interactive data analysis and model building.

## How It Works
1. **Import Dependencies:** Import necessary libraries for data manipulation, visualization, and modeling.
2. **Data Collection & Analysis:** Load the insurance dataset and explore its structure, including checking for missing values and basic statistical measures.
3. **Data Visualization:** Plot various features like age, gender, BMI, number of children, smoker status, and region to understand their distributions and relationships.
4. **Data Preprocessing:** Encode categorical variables (e.g., sex, smoker, region) into numerical values. Split the data into features (X) and target variable (Y).
5. **Train-Test Split:** Divide the dataset into training and testing sets.
6. **Model Training:** Train a linear regression model on the training data.
7. **Model Evaluation:** Evaluate the model's performance on both training and testing data using the R-squared metric.
8. **Predictive System:** Build a system to predict insurance costs for new data inputs.

## Applications
- Predict insurance costs based on individual attributes.
- Understand factors that influence insurance costs.

## Example Code
```python
# Importing the Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Data Collection & Analysis
# loading the data from csv file to a Pandas DataFrame
insurance_dataset = pd.read_csv('/content/insurance.csv')

# first 5 rows of the dataframe
insurance_dataset.head()

# number of rows and columns
insurance_dataset.shape

# getting some information about the dataset
insurance_dataset.info()

# checking for missing values
insurance_dataset.isnull().sum()

# statistical Measures of the dataset
insurance_dataset.describe()

# distribution of age value
sns.set()
plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['age'])
plt.title('Age Distribution')
plt.show()

# Gender column
plt.figure(figsize=(6,6))
sns.countplot(x='sex', data=insurance_dataset)
plt.title('Sex Distribution')
plt.show()

# BMI distribution
plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['bmi'])
plt.title('BMI Distribution')
plt.show()

# children column
plt.figure(figsize=(6,6))
sns.countplot(x='children', data=insurance_dataset)
plt.title('Children')
plt.show()

# smoker column
plt.figure(figsize=(6,6))
sns.countplot(x='smoker', data=insurance_dataset)
plt.title('Smoker')
plt.show()

# region column
plt.figure(figsize=(6,6))
sns.countplot(x='region', data=insurance_dataset)
plt.title('Region')
plt.show()

# distribution of charges value
plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['charges'])
plt.title('Charges Distribution')
plt.show()

# Data Pre-Processing
# Encoding the categorical features
# encoding sex column
insurance_dataset.replace({'sex':{'male':0,'female':1}}, inplace=True)

# encoding 'smoker' column
insurance_dataset.replace({'smoker':{'yes':0,'no':1}}, inplace=True)

# encoding 'region' column
insurance_dataset.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}, inplace=True)

# Splitting the Features and Target
X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']

# Splitting the data into Training data & Testing Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Model Training
# Linear Regression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Model Evaluation
# prediction on training data
training_data_prediction = regressor.predict(X_train)

# R squared value
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R squared value: ', r2_train)

# prediction on test data
test_data_prediction = regressor.predict(X_test)

# R squared value
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R squared value: ', r2_test)

# Building a Predictive System
input_data = (31,1,25.74,0,1,0)

# changing input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = regressor.predict(input_data_reshaped)
print('The insurance cost is USD ', prediction[0])
