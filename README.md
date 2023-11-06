# EXPERIMENT-07

# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1. PREPARE YOUR DATA:

Collect and clean data on employee salaries and features


Split data into training and testing sets

2.DEFINE YOUR MODEL 

Use a Decision Tree Regressor to recursively partition data based on input features

Determine maximum depth of tree and other hyperparameters

3. TRAIN YOUR MODEL:

Fit model to training data

Calculate mean salary value for each subset

4. EVALUATE YOUR MODEL:

Use model to make predictions on testing data

Calculate metrics such as MAE and MSE to evaluate performance

5.TUNE HYPERPARAMETERS:

Experiment with different hyperparameters to improve performance

6.DEPLOY YOUR MODEL:

Use model to make predictions on new data in real-world application.

## Program:
```py

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: MIDHUN AZHAHU RAJA P
RegisterNumber: 212222240066

import pandas as pd
data = pd.read_csv("dataset/Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()

x = data[["Position", "Level"]]
x.head()

y = data["Salary"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 2)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
mse = metrics.mean_squared_error(y_test, y_pred)
mse

r2 = metrics.r2_score(y_test, y_pred)
r2

dt.predict([[5, 6]])
```

## Output:
1.INITIAL DATASET:

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393818/b05eb46f-f6a2-4f6b-92dd-7c04151699d4)

2.DATA INFO:

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393818/d8620361-291d-4837-b591-8dcd2d6c11e9)

3.OPTIMIZATION OF NULL VALUES:

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393818/6955af0c-f854-4239-b2b2-5415f5dc30ff)

4.CONVERTINH STRING LITERALS TO NUMERICAL VALUES USING LABEL ENCODER

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393818/11662a2c-56f1-4ac6-84b6-eb2501c4700b)

5.ASSIGNING X AND Y VALUES:

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393818/b46eb74f-e262-4d4b-becf-bccbcf036fdc)

6.MEAN SQUARED ERROR:

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393818/41de82f6-062c-4d38-908f-2f3cdf414352)

7.R2 VARIANCE:

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393818/78cc6ba3-87b1-4a80-b526-a05090ac28b9)

8.PREDICTION:

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393818/0a407110-fe10-48b7-8915-af1d8a67fe25)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
