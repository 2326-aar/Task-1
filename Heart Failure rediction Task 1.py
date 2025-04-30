# Step 1: Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats

# Step 2: Load Dataset
df = pd.read_csv("/content/heart Failure prediction Dataset.csv")
print(df.head())
print(df.info())

# Step 3: Check and Handle Missing Values
print("\nMissing Values:\n", df.isnull().sum())

# Step 4: Encode Categorical Features
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['ExerciseAngina'] = le.fit_transform(df['ExerciseAngina'])

# One-hot encoding for other categorical columns
df = pd.get_dummies(df, columns=['ChestPainType', 'RestingECG', 'ST_Slope'], drop_first=True)

# Step 5: Normalize / Standardize Numeric Features
scaler = StandardScaler()
num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
df[num_cols] = scaler.fit_transform(df[num_cols])

# Step 6: Visualize Outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[num_cols])
plt.title("Boxplot to Detect Outliers")
plt.show()

# Step 7: Remove Outliers using Z-Score
df = df[(np.abs(stats.zscore(df[num_cols])) < 3).all(axis=1)]

# Step 8: Final Data Overview
print("\nCleaned Data Preview:\n", df.head())
print("\nData Shape After Cleaning:", df.shape)
