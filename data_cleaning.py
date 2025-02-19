import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define file path (since the CSV is in the same folder)
file_path = r"C:\Users\pooji\OneDrive\Desktop\git hub projects\customer churn prediction\teleco_churn.csv"


# Load the dataset
print("Loading dataset...")
df = pd.read_csv(file_path, encoding='utf-8')  # Read CSV

# Print first 5 rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Checking for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())


#cleaning the dataset by handling the missing values 

# Fill missing values in 'TotalCharges' with 0 (or median)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')  # Convert to numeric
df.fillna({'TotalCharges': df['TotalCharges'].median()}, inplace=True)  # Fill missing with median

# Drop any remaining missing values
df.dropna(inplace=True)

#Converting Categorical data to Numerical values for making it machine learning algorithms easier
# Convert categorical 'Yes/No' values to binary (1/0)
binary_columns = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity',
                  'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']

for col in binary_columns:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

 # Convert 'gender' column to binary (0: Female, 1: Male)
df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})

# Convert categorical features into numerical values using one-hot encoding
df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'], drop_first=True)

# Drop unnecessary columns
df.drop(['customerID'], axis=1, inplace=True)

#saving the modified and data cleaned dataset 

df.to_csv("cleaned_teleco_churn.csv", index=False)
print("Cleaned dataset saved as 'cleaned_teleco_churn.csv'")

#exploratory Data Analysis(EDA)

#datatypes checking
print("\nColumn Data Types:")
print(df.dtypes)

#Churn Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.show()

#Correlation Analysis
plt.figure(figsize=(12,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()