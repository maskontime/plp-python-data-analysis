# 📊 Analyzing Data with Pandas and Visualizing Results with Matplotlib
# This script demonstrates how to load, analyze, and visualize a dataset
# using pandas and matplotlib (with seaborn for better styling).
# We will use the classic Iris dataset.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
iris_data = load_iris()

# Convert into pandas DataFrame
df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)

# Add species column
df['species'] = iris_data.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Display first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Check info and missing values
print("\nDataset Info:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Group by species and compute mean values
print("\nMean values by species:")
print(df.groupby('species').mean())

# Line Chart
plt.figure(figsize=(8,5))
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length', color='blue')
plt.title("Line Chart of Sepal Length")
plt.xlabel("Index (like time)")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.show()

# Bar Chart
plt.figure(figsize=(8,5))
sns.barplot(x='species', y='petal length (cm)', data=df, ci=None)
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# Histogram
plt.figure(figsize=(8,5))
plt.hist(df['sepal width (cm)'], bins=15, color='green', edgecolor='black')
plt.title("Histogram of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# Scatter Plot
plt.figure(figsize=(8,5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.show()

# Findings & Observations
print("\nFindings & Observations:")
print("1. Setosa has smaller petals compared to the other two species.")
print("2. Virginica generally has the longest petals and sepals.")
print("3. The scatter plot shows clear separation between Setosa and the other two species.")
print("4. Sepal width is concentrated between 2.5–3.5 cm.")
