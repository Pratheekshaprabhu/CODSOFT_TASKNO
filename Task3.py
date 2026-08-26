# CODSOFT DATA ANALYTICS INTERNSHIP
# TASK 3 - DATA VISUALIZATION

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_style("whitegrid")

# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

df = sns.load_dataset("titanic")

print("Dataset loaded successfully!")
print(df.head())

# --------------------------------------------------
# 2. BAR CHART - SURVIVAL BY GENDER
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(data=df, x="sex", hue="survived")

plt.title("Survival Count by Gender", fontsize=16)
plt.xlabel("Gender", fontsize=12)
plt.ylabel("Number of Passengers", fontsize=12)
plt.legend(title="Survived", labels=["No", "Yes"])

plt.tight_layout()
plt.show()

# --------------------------------------------------
# 3. LINE CHART - AVERAGE FARE BY CLASS
# --------------------------------------------------

average_fare = df.groupby("pclass")["fare"].mean().reset_index()

plt.figure(figsize=(8, 5))

sns.lineplot(
    data=average_fare,
    x="pclass",
    y="fare",
    marker="o",
    linewidth=2
)

plt.title("Average Fare by Passenger Class", fontsize=16)
plt.xlabel("Passenger Class", fontsize=12)
plt.ylabel("Average Fare", fontsize=12)

plt.xticks([1, 2, 3])

plt.tight_layout()
plt.show()

# --------------------------------------------------
# 4. PIE CHART - SURVIVAL DISTRIBUTION
# --------------------------------------------------

survival_count = df["survived"].value_counts()

plt.figure(figsize=(7, 7))

plt.pie(
    survival_count,
    labels=["Not Survived", "Survived"],
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Overall Survival Distribution", fontsize=16)

plt.show()

# --------------------------------------------------
# 5. HISTOGRAM - AGE DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="age",
    bins=30,
    kde=True
)

plt.title("Age Distribution of Passengers", fontsize=16)
plt.xlabel("Age", fontsize=12)
plt.ylabel("Number of Passengers", fontsize=12)

plt.tight_layout()
plt.show()

# --------------------------------------------------
# 6. SCATTER PLOT - AGE VS FARE
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="age",
    y="fare",
    hue="survived"
)

plt.title("Age vs Fare", fontsize=16)
plt.xlabel("Age", fontsize=12)
plt.ylabel("Fare", fontsize=12)
plt.legend(title="Survived", labels=["No", "Yes"])

plt.tight_layout()
plt.show()

# --------------------------------------------------
# 7. ADDITIONAL VISUALIZATION
# SURVIVAL RATE BY PASSENGER CLASS
# --------------------------------------------------

survival_rate = df.groupby("pclass")["survived"].mean() * 100

plt.figure(figsize=(8, 5))

sns.barplot(
    x=survival_rate.index,
    y=survival_rate.values
)

plt.title("Survival Rate by Passenger Class", fontsize=16)
plt.xlabel("Passenger Class", fontsize=12)
plt.ylabel("Survival Rate (%)", fontsize=12)

plt.tight_layout()
plt.show()

# --------------------------------------------------
# 8. ADDITIONAL VISUALIZATION
# FARE DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="fare",
    bins=30,
    kde=True
)

plt.title("Fare Distribution", fontsize=16)
plt.xlabel("Fare", fontsize=12)
plt.ylabel("Number of Passengers", fontsize=12)

plt.tight_layout()
plt.show()

# --------------------------------------------------
# 9. CORRELATION HEATMAP
# --------------------------------------------------

numeric_data = df.select_dtypes(include="number")

plt.figure(figsize=(10, 7))

sns.heatmap(
    numeric_data.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Between Numerical Variables", fontsize=16)

plt.tight_layout()
plt.show()

# --------------------------------------------------
# 10. KEY INSIGHTS
# --------------------------------------------------

print("\n========== KEY INSIGHTS ==========")

print("1. Female passengers had a higher survival rate than male passengers.")

print("2. First-class passengers had a higher survival rate than second- and third-class passengers.")

print("3. Most passengers were within a broad adult age range.")

print("4. Fare values were highly uneven, with some passengers paying very high fares.")

print("5. The scatter plot shows the relationship between passenger age and fare.")

print("6. Visualization makes it easier to identify patterns and differences in the dataset.")

print("\nTASK 3 COMPLETED SUCCESSFULLY!")