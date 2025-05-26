# Task 1: Data Cleaning & Preprocessing using Titanic Dataset
# Author: Hema 
# Date: 26-05-2025

# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Step 1: Loading the dataset
df = pd.read_csv(r'C:\Users\Lenovo\Downloads\Titanic-Dataset (1).csv')
# Step 2: Exploring basic info about the dataset
print("Basic Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# Step 3: Checking for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Step 4: Handling missing values
# For numerical columns: fill NA with median
# For categorical columns: fill NA with most frequent value (mode)
numeric_features = ['Age', 'Fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['Pclass', 'Sex', 'Embarked']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Step 5: Apply transformations to selected features
X = df[['Age', 'Fare', 'Pclass', 'Sex', 'Embarked']]
y = df['Survived']  # Target variable

X_preprocessed = preprocessor.fit_transform(X)

# Optional: Convert sparse matrix to DataFrame for better readability
X_processed_df = pd.DataFrame(X_preprocessed.toarray())
print("\nPreprocessed Data Sample:")
print(X_processed_df.head())

# Step 6: Detecting outliers in 'Age' and 'Fare' using boxplots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(x=df['Age'])
plt.title('Age Outliers')

plt.subplot(1, 2, 2)
sns.boxplot(x=df['Fare'])
plt.title('Fare Outliers')

plt.tight_layout()
plt.savefig('outliers_before_removal.png')  # Save plot
plt.show()

# Step 7: Remove outliers using IQR method
def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

df_cleaned = remove_outliers(df, 'Age')
df_cleaned = remove_outliers(df_cleaned, 'Fare')

# Show outliers after removal
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x=df_cleaned['Age'])
plt.title('Age After Outlier Removal')

plt.subplot(1, 2, 2)
sns.boxplot(x=df_cleaned['Fare'])
plt.title('Fare After Outlier Removal')

plt.tight_layout()
plt.savefig('outliers_after_removal.png')  # Save updated plot
plt.show()

# Final cleaned data
print("\nCleaned Data Shape:", df_cleaned.shape)

# You can now proceed to train models or save the cleaned data
df_cleaned.to_csv('cleaned_titanic.csv', index=False)
print("\nCleaned data saved as 'cleaned_titanic.csv'")