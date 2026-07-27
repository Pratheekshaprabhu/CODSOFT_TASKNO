import pandas as pd


df = pd.read_csv("employees.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

df = df.drop_duplicates()

df["Age"] = df["Age"].fillna(df["Age"].mean())

df["Salary"] = df["Salary"].fillna(df["Salary"].mean())

df["Age"] = df["Age"].astype(int)

df.to_csv("cleaned_employees.csv", index=False)

print("\nCleaned Dataset:")
print(df)

print("\nCleaned dataset saved as cleaned_employees.csv")