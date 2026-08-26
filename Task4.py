# ============================================================
# CODSOFT DATA ANALYTICS INTERNSHIP
# TASK 4 - CUSTOMER SEGMENTATION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# ============================================================
# 1. CREATE CUSTOMER DATASET
# ============================================================

data = {
    "CustomerID": range(1, 21),

    "Gender": [
        "Male", "Female", "Female", "Male", "Female",
        "Female", "Male", "Female", "Male", "Female",
        "Male", "Female", "Female", "Male", "Male",
        "Female", "Female", "Male", "Female", "Male"
    ],

    "Age": [
        19, 21, 20, 23, 31,
        22, 35, 23, 64, 30,
        67, 35, 58, 24, 37,
        22, 35, 20, 52, 35
    ],

    "Annual_Income": [
        15, 16, 17, 18, 19,
        20, 21, 22, 23, 24,
        25, 26, 27, 28, 29,
        30, 31, 32, 33, 34
    ],

    "Spending_Score": [
        39, 81, 6, 77, 40,
        76, 6, 94, 3, 72,
        14, 99, 15, 77, 13,
        79, 35, 66, 29, 98
    ]
}

df = pd.DataFrame(data)

print("\nFirst 5 Customers:")
print(df.head())

# ============================================================
# 2. BASIC INFORMATION
# ============================================================

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDescriptive Statistics:")
print(df.describe())

# ============================================================
# 3. CUSTOMER DISTRIBUTION BY GENDER
# ============================================================

gender_count = df["Gender"].value_counts()

print("\nCustomers by Gender:")
print(gender_count)

plt.figure(figsize=(7, 5))

sns.barplot(
    x=gender_count.index,
    y=gender_count.values
)

plt.title("Customer Distribution by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

# ============================================================
# 4. AGE DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Age"],
    bins=10,
    kde=True
)

plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

# ============================================================
# 5. ANNUAL INCOME DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Annual_Income"],
    bins=10,
    kde=True
)

plt.title("Annual Income Distribution")
plt.xlabel("Annual Income")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

# ============================================================
# 6. INCOME VS SPENDING SCORE
# ============================================================

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="Annual_Income",
    y="Spending_Score",
    hue="Gender",
    s=100
)

plt.title("Annual Income vs Spending Score")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.tight_layout()
plt.show()

# ============================================================
# 7. AGE VS SPENDING SCORE
# ============================================================

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="Age",
    y="Spending_Score",
    hue="Gender",
    s=100
)

plt.title("Age vs Spending Score")
plt.xlabel("Age")
plt.ylabel("Spending Score")

plt.tight_layout()
plt.show()

# ============================================================
# 8. CUSTOMER SEGMENTATION
# ============================================================

def customer_segment(score):

    if score >= 70:
        return "High Value"

    elif score >= 40:
        return "Medium Value"

    else:
        return "Low Value"


df["Customer_Segment"] = df["Spending_Score"].apply(
    customer_segment
)

print("\nCustomer Segments:")
print(df["Customer_Segment"].value_counts())

# ============================================================
# 9. CUSTOMER SEGMENT VISUALIZATION
# ============================================================

segment_count = df["Customer_Segment"].value_counts()

plt.figure(figsize=(8, 5))

sns.barplot(
    x=segment_count.index,
    y=segment_count.values
)

plt.title("Customer Segmentation")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

# ============================================================
# 10. AVERAGE SPENDING BY GENDER
# ============================================================

gender_spending = df.groupby("Gender")["Spending_Score"].mean()

print("\nAverage Spending Score by Gender:")
print(gender_spending)

plt.figure(figsize=(7, 5))

sns.barplot(
    x=gender_spending.index,
    y=gender_spending.values
)

plt.title("Average Spending Score by Gender")
plt.xlabel("Gender")
plt.ylabel("Average Spending Score")

plt.tight_layout()
plt.show()

# ============================================================
# 11. AVERAGE INCOME BY CUSTOMER SEGMENT
# ============================================================

segment_income = df.groupby(
    "Customer_Segment"
)["Annual_Income"].mean()

print("\nAverage Income by Customer Segment:")
print(segment_income)

# ============================================================
# 12. PIE CHART
# ============================================================

plt.figure(figsize=(7, 7))

plt.pie(
    segment_count.values,
    labels=segment_count.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Customer Segment Distribution")

plt.show()

# ============================================================
# 13. CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(8, 6))

correlation = df[
    ["Age", "Annual_Income", "Spending_Score"]
].corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Customer Data Correlation")

plt.tight_layout()
plt.show()

# ============================================================
# 14. MOST VALUABLE CUSTOMER GROUP
# ============================================================

valuable_segment = df.groupby(
    "Customer_Segment"
)["Spending_Score"].mean().idxmax()

valuable_score = df.groupby(
    "Customer_Segment"
)["Spending_Score"].mean().max()

print("\n========================================")
print("       MOST VALUABLE CUSTOMER GROUP")
print("========================================")

print("Customer Group:", valuable_segment)
print("Average Spending Score:", round(valuable_score, 2))

# ============================================================
# 15. BUSINESS INSIGHTS
# ============================================================

print("\n========================================")
print("             KEY INSIGHTS")
print("========================================")

print("""
1. Customers can be divided into Low, Medium and High Value
   segments based on their spending score.

2. High-value customers have the highest spending scores.

3. Annual income and spending score can be compared to
   identify potential high-value customers.

4. Age distribution helps identify the major customer age groups.

5. Gender-based analysis helps understand differences in
   customer purchasing behavior.

6. Scatter plots help identify relationships between income,
   age and spending score.
""")

# ============================================================
# 16. MARKETING STRATEGIES
# ============================================================

print("\n========================================")
print("        MARKETING STRATEGIES")
print("========================================")

print("""
1. High Value Customers:
   Offer loyalty rewards, exclusive discounts and premium products.

2. Medium Value Customers:
   Use personalized offers and product recommendations
   to increase their spending.

3. Low Value Customers:
   Provide attractive introductory offers and discounts
   to encourage purchases.

4. High Income Customers:
   Promote premium products and special membership programs.

5. Young Customers:
   Use social media campaigns and digital promotions.

6. Repeat Customers:
   Introduce loyalty points and reward programs.
""")

print("\nTASK 4 COMPLETED SUCCESSFULLY!")