# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement Logistic Regression for classifying food choices based on nutritional information.
Developed by: R.Mushafina
RegisterNumber: 212224220067
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import LabelEncoder, MinMaxScaler 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report 
import seaborn as sns 
import matplotlib.pyplot as plt
# Load the dataset 
df=pd.read_csv("C:/Users/admin/Downloads/food_items (1).csv")
# Inspect the dataset
print('Name: R.Mushafina')
print('Reg.No: 212224220067')
print("Dataset Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info())
x_raw=df.iloc[:,:-1]
y_raw=df.iloc[:,-1:]
scaler = MinMaxScaler()
# Scaling the raw input features 
x=scaler.fit_transform(x_raw)
# Create a LabEncoder object
label_encoder=LabelEncoder()
# Encode the target variable 
y = label_encoder.fit_transform(y_raw.values.ravel())
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=123)
# L2 penalty to shrink coefficients 
penalty='l2'
#Our classificiation problem is multinomial 
multi_class='multinomial'
# Use lbfgs for L2 penalty and multinomial classes 
solver='lbfgs'
# Max iteration = 1000
max_iter=1000
# Define logistic regression model with above arguments 
l2_model = LogisticRegression(random_state=123,penalty=penalty,multi_class=multi_class, solver=solver, max_iter=max_iter)
l2_model.fit(x_train,y_train)
y_pred=l2_model.predict(x_test)
# Evaluate the model 
print('Name: R.Mushafina')
print('Reg.No: 212224220067')
print("\nModel Evaluation:")
print("Accuracy:",accuracy_score(y_test,y_pred))
print("\nClassification Report:")
print(classification_report(y_test,y_pred))

# Confusion Matrix 
conf_matrix=confusion_matrix(y_test,y_pred)
print(conf_matrix)
*/
```

## Output:
<img width="933" height="753" alt="image" src="https://github.com/user-attachments/assets/9639fcbb-a58c-4692-8733-51dcdac3c955" />
<img width="871" height="764" alt="image" src="https://github.com/user-attachments/assets/527f0800-9ad4-426d-84aa-e19a86232994" />
<img width="834" height="691" alt="image" src="https://github.com/user-attachments/assets/d77b3cfb-f92b-45c8-8130-822c1dfa081b" />




## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
