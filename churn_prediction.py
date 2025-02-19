import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#loading the dataset
file_path = "cleaned_teleco_churn.csv"
df = pd.read_csv(file_path)

#Defining and assigning the features (x) and target variable (y)
X = df.drop(columns=['Churn'])  # All columns except 'Churn'
y = df['Churn']  # Target variable

print("\nChecking for missing values before training:")
print(df.isnull().sum())

#spliting the dataset into the training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fill missing values with column median (numerical) or mode (categorical)
df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

print("\nChecking for NaN values in X_train before training:")
print(pd.DataFrame(X_train).isnull().sum())

import pandas as pd

# Convert NumPy array back to DataFrame
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

# Fill missing values
X_train.fillna(X_train.median(numeric_only=True), inplace=True)
X_train.fillna(X_train.mode().iloc[0], inplace=True)

X_test.fillna(X_test.median(numeric_only=True), inplace=True)
X_test.fillna(X_test.mode().iloc[0], inplace=True)

# Verify missing values again
print("\n Missing values in X_train after filling:")
print(X_train.isnull().sum())  # This should be all 0s


# 1.Train a Logistic Regression Model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

#Make Predictions
y_pred_log = log_reg.predict(X_test)

#Evaluate Model
print("\n📊 Logistic Regression Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

#2.Train a Decision Tree Model
from sklearn.tree import DecisionTreeClassifier

# Train Decision Tree model (Ensure this part is in your script)
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

#Make Predictions
y_pred_tree = decision_tree.predict(X_test)

#Evaluate Model
print("\n🌲 Decision Tree Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))

#Plot Confusion Matrices
fig, ax = plt.subplots(1, 2, figsize=(12, 5))


# Logistic Regression Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, fmt="d", cmap="Blues", ax=ax[0])
ax[0].set_title("Logistic Regression Confusion Matrix")

# Decision Tree Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_tree), annot=True, fmt="d", cmap="Greens", ax=ax[1])
ax[1].set_title("Decision Tree Confusion Matrix")

plt.pause(3)  
plt.close() 

print("Saving models...")  # Debugging print statement

# Save the trained models
joblib.dump(log_reg, "logistic_regression_model.pkl")
joblib.dump(decision_tree, "decision_tree_model.pkl")

print("\nModels saved successfully as 'logistic_regression_model.pkl' & 'decision_tree_model.pkl'")
