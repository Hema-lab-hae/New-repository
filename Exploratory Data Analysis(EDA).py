# Step 1: Install required libraries (run this only if needed)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Loading the Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Step 3: Displayng the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Step 4: Data Information and Summary
print("\nDataset Information:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# Step 5: Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 6: Visualizations
# Histogram of Age
plt.figure(figsize=(6,4))
sns.histplot(df['Age'].dropna(), kde=True)
plt.title('Age Distribution')
plt.show()

# Boxplot of Fare
plt.figure(figsize=(6,4))
sns.boxplot(x=df['Fare'])
plt.title('Fare Boxplot')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,8))
correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Survival Count by Gender
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival Count by Gender')
plt.show()
