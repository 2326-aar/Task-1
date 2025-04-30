# Task-1
# 🫀 Heart Disease Dataset - Data Cleaning & Preprocessing (AI & ML Internship Task 1)

This repository contains the code and process, which focuses on data cleaning and preprocessing using a real-world dataset. The dataset used for this task is the [Heart Disease Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).

---

## 📁 Dataset Information

- **Dataset**: Heart Disease Prediction Dataset
- **Source**: [Kaggle - fedesoriano](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Size**: 918 samples × 12 features
- **Target Variable**: `HeartDisease` (1 = presence of heart disease, 0 = absence)

---

## 🔧 Tools & Libraries Used

- Python 3.x
- [Pandas](https://pandas.pydata.org/) – for data manipulation
- [NumPy](https://numpy.org/) – for numerical operations
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) – for data visualization
- [Scikit-learn](https://scikit-learn.org/) – for preprocessing and scaling

---

## 🔍 Preprocessing Steps

### 1. Data Import & Exploration
- Loaded dataset into a Pandas DataFrame
- Checked data types, missing values, and value distributions

### 2. Handling Missing Values
- Verified the dataset had no missing values (if any, fill using `.fillna()`)

### 3. Encoding Categorical Variables
- Label Encoded binary categorical columns (`Sex`, `ExerciseAngina`)
- One-Hot Encoded multi-class columns (`ChestPainType`, `RestingECG`, `ST_Slope`) using `pd.get_dummies()`

### 4. Feature Scaling
- Used `StandardScaler()` from scikit-learn to scale numerical features like `Age`, `RestingBP`, `Cholesterol`, `MaxHR`, and `Oldpeak`

### 5. Outlier Detection & Removal
- Visualized outliers using boxplots
- Removed extreme values using Z-score method (`zscore < 3`)

---

## 📈 Results

After preprocessing:
- Dataset shape reduced slightly after outlier removal
- Features were converted into numerical and scaled appropriately
- Ready for training a machine learning model

---
