# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and separate the feature (Level) and target variable (Salary).

2.Train the Decision Tree Regressor model using the training data.

3.Use the trained model to predict salary for the dataset or a new level value.

4.Visualize the results by plotting the actual data points and the predicted regression curve.
 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: KARTHICK BALAJI
RegisterNumber: 212225040174 
*/
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# ------------------------------
# Step 1: Sample dataset
# ------------------------------
data = {
    'Position': ['Business Analyst', 'Junior Consultant', 'Senior Consultant',
                 'Manager', 'Country Manager', 'Region Manager',
                 'Partner', 'Senior Partner', 'C-level', 'CEO'],
    'Level': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [45000, 50000, 60000, 80000, 110000, 150000, 200000, 300000, 500000, 1000000]
}

df = pd.DataFrame(data)

# ------------------------------
# Step 2: Split features and target
# ------------------------------
X = df[['Level']]     # Feature (Level)
y = df['Salary']      # Target (Salary)

# ------------------------------
# Step 3: Create Decision Tree Regressor
# ------------------------------
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X, y)

# ------------------------------
# Step 4: Predict salary for the dataset or new levels
# ------------------------------
y_pred = regressor.predict(X)
print("Predicted salaries:", y_pred)

# Example: predict salary for a new employee at level 6.5
level = np.array([[6.5]])
predicted_salary = regressor.predict(level)
print(f"Predicted Salary for level {level[0][0]}: {predicted_salary[0]}")

# ------------------------------
# Step 5: Visualize the results (High-resolution curve)
# ------------------------------
X_grid = np.arange(min(X.values), max(X.values)+0.01, 0.01)  # High-resolution for smoother curve
X_grid = X_grid.reshape(-1, 1)

plt.scatter(X, y, color='red', label='Actual Salary')
plt.plot(X_grid, regressor.predict(X_grid), color='blue', label='Decision Tree Prediction')
plt.title('Decision Tree Regression: Level vs Salary')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.legend()
plt.show()

```

## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)
<img width="1040" height="73" alt="image" src="https://github.com/user-attachments/assets/ab7a0028-7f9f-454b-9c92-9ef95d695c01" />
<Figure size 640x480 with 1 Axes><img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/661d8c67-5b93-445a-b372-4f1f3ea1f5c5" />






## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
