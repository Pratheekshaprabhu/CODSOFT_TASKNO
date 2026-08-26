# CODSOFT DATA ANALYTICS INTERNSHIP
# TASK 2 - Exploratory Data Analysis

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

# Load Titanic dataset
df = sns.load_dataset("titanic")

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# --------------------------------------------------
# 2. EXAMINE DATASET
# --------------------------------------------------

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# --------------------------------------------------
# 3. DESCRIPTIVE STATISTICS
# --------------------------------------------------

print("\nDescriptive Statistics:")
print(df.describe())

# Mean
print("\nMean:")
print(df.select_dtypes(include=np.number).mean())

# Median
print("\nMedian:")
print(df.select_dtypes(include=np.number).median())

# Standard Deviation
print("\nStandard Deviation:")
print(df.select_dtypes(include=np.number).std())

# --------------------------------------------------
# 4. SURVIVAL ANALYSIS
# --------------------------------------------------

print("\nOverall Survival Rate:")
print(df["survived"].mean() * 100)

print("\nSurvival Count:")
print(df["survived"].value_counts())

# Survival by gender
print("\nSurvival Rate by Gender:")
print(df.groupby("sex")["survived"].mean() * 100)

# Survival by passenger class
print("\nSurvival Rate by Passenger Class:")
print(df.groupby("pclass")["survived"].mean() * 100)

# --------------------------------------------------
# 5. DISTRIBUTION OF AGE
# --------------------------------------------------

plt.figure(figsize=(8, 5))
sns.histplot(df["age"].dropna(), bins=30, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.show()

# --------------------------------------------------
# 6. FARE DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(8, 5))
sns.histplot(df["fare"], bins=30, kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")
plt.show()

# --------------------------------------------------
# 7. SURVIVAL BY GENDER
# --------------------------------------------------

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="sex", hue="survived")
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

# --------------------------------------------------
# 8. SURVIVAL BY PASSENGER CLASS
# --------------------------------------------------

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="pclass", hue="survived")
plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

# --------------------------------------------------
# 9. AGE VS FARE RELATIONSHIP
# --------------------------------------------------

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="age", y="fare", hue="survived")
plt.title("Relationship Between Age and Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()

# --------------------------------------------------
# 10. CORRELATION ANALYSIS
# --------------------------------------------------

numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(10, 7))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# --------------------------------------------------
# 11. OUTLIER DETECTION - AGE
# --------------------------------------------------

plt.figure(figsize=(7, 5))
sns.boxplot(y=df["age"])
plt.title("Age Outliers")
plt.ylabel("Age")
plt.show()

# --------------------------------------------------
# 12. OUTLIER DETECTION - FARE
# --------------------------------------------------

plt.figure(figsize=(7, 5))
sns.boxplot(y=df["fare"])
plt.title("Fare Outliers")
plt.ylabel("Fare")
plt.show()

# --------------------------------------------------
# 13. IQR METHOD FOR FARE OUTLIERS
# --------------------------------------------------

Q1 = df["fare"].quantile(0.25)
Q3 = df["fare"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = df[(df["fare"] < lower_limit) | (df["fare"] > upper_limit)]

print("\nFare Outlier Analysis:")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)
print("Number of Fare Outliers:", len(outliers))

# --------------------------------------------------
# 14. BUSINESS QUESTIONS
# --------------------------------------------------

print("\n========== BUSINESS QUESTIONS ==========")

# Question 1
highest_gender = df.groupby("sex")["survived"].mean().idxmax()
highest_gender_rate = df.groupby("sex")["survived"].mean().max() * 100

print("\n1. Which gender had the highest survival rate?")
print(highest_gender, "-", round(highest_gender_rate, 2), "%")

# Question 2
highest_class = df.groupby("pclass")["survived"].mean().idxmax()
highest_class_rate = df.groupby("pclass")["survived"].mean().max() * 100

print("\n2. Which passenger class had the highest survival rate?")
print("Class", highest_class, "-", round(highest_class_rate, 2), "%")

# Question 3
average_fare = df["fare"].mean()

print("\n3. What was the average fare?")
print(round(average_fare, 2))

# Question 4
average_age = df["age"].mean()

print("\n4. What was the average passenger age?")
print(round(average_age, 2))

# Question 5
print("\n5. Are there outliers in fare?")
print("Yes, outliers were detected using the IQR method.")

# --------------------------------------------------
# 15. SHORT REPORT
# --------------------------------------------------

print("\n========== SHORT REPORT ==========")

print("""
Titanic Dataset - Exploratory Data Analysis Report

1. The dataset contains information about passengers,
   including age, gender, passenger class, fare and survival.

2. The analysis shows that survival rates were different
   between male and female passengers.

3. Passenger class had a strong relationship with survival.
   Higher-class passengers generally had better survival rates.

4. Age values show a wide distribution across passengers.

5. Fare has a highly uneven distribution, with some passengers
   paying significantly higher fares than most passengers.

6. Boxplot and IQR analysis identified several fare outliers.

7. The correlation analysis helps identify relationships
   between numerical variables.

8. Overall, gender, passenger class and fare are important
   variables for understanding survival patterns in the dataset.
""")

print("\nTask 2 Completed Successfully!")