<h1>
  <span class="headline">ML Workflow and Best Practices</span>
  <span class="subhead">Exploratory Data Analysis</span>
</h1>

**Learning objective:** By the end of this lesson, you will be able to perform exploratory data analysis on preprocessed data.

## An Introduction to Exploratory Data Analysis

Exploratory Data Analysis (EDA) is the process of analyzing datasets to summarize their main characteristics. EDA helps in understanding the data distribution, relationships between variables, and patterns that might affect the model’s performance. This step often involves visualizations (histograms, scatter plots, box plots) and statistical summaries (mean, variance, correlations). The goals of EDA is to is to understand the data's structure and provide insights about the data in the context of:

- 🧮 **Understanding Data Distributions**  
  - Assess the spread, central tendency, and variability of numerical data.  
  - Identify skewness, kurtosis, and statistical properties.

- 🔗 **Identifying Relationships**  
  - Explore interactions and dependencies between variables.  
  - Highlight correlations that may inform model features.

- 🚨 **Detecting Anomalies**  
  - Spot outliers or unusual patterns that could distort model predictions.

- 🔍 **Assessing Data Quality**  
  - Check for missing values, duplicates, and inconsistencies.

- 🛠️ **Guiding Feature Selection**  
  - Determine which features are most relevant for the problem.

## Common Techniques in EDA

There's no better way to learn EDA techniques than doing it. For this purpose, lets consider a small dataset containing student performance data:

| Student_ID | Gender | Math_Score | Reading_Score | Writing_Score | Hours_Studied | Extra_Curricular |
| ---------- | ------ | ---------- | ------------- | ------------- | ------------- | ---------------- |
| 1          | Male   | 85         | 78            | 80            | 6             | Yes              |
| 2          | Female | 92         | 88            | 90            | 8             | No               |
| 3          | Male   | 78         | 74            | 75            | 5             | Yes              |
| 4          | Female | 70         | 65            | 68            | 4             | No               |
| 5          | Male   | 60         | 58            | 55            | 2             | No               |


In order to use this dataset to learn the EDA techniques using python, we can store it in a CSV file called  `students.csv` as shown below. Lets save this CSV file in the same folder where our jupyter notebook resides.

```csv
Student_ID,Gender,Math_Score,Reading_Score,Writing_Score,Hours_Studied,Extra_Curricular
1,Male,85,78,80,6,Yes
2,Female,92,88,90,8,No
3,Male,78,74,75,5,Yes
4,Female,70,65,68,4,No
5,Male,60,58,55,2,No
```
As a final step before we dive in, lets import the required libraries...
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import StandardScaler
```
... and load our dataset (CSV file).
```python
df = pd.read_csv("students.csv")
```
Now, lets dive in to the various EDA techniques.
### Univariate Analysis
Univariate analysis focuses on a single variable, analyzing its distribution, central tendency, and spread. There are various types of univariate analysis that could be performed, namely:

#### 1. Categorical Features Analysis
Categorical variables represent groups or labels. Common techniques include frequency tables and bar plots.

```python

# Count unique values in categorical features
print(df["Gender"].value_counts())
print(df["Extra_Curricular"].value_counts())

# Visualize categorical features
sns.countplot(x="Gender", data=df)
plt.title("Distribution of Gender")
plt.show()
```

#### 2. Numerical Features Analysis
Numerical variables include continuous values such as scores and hours studied.

```python
# Summary statistics
print(df[["Math_Score", "Reading_Score", "Writing_Score"]].describe())

# Histogram to visualize distribution
df["Math_Score"].hist(bins=5)
plt.title("Distribution of Math Scores")
plt.xlabel("Scores")
plt.ylabel("Frequency")
plt.show()
```
#### 3. Missing Value Detection
We can use the `isnull()` method to detect missing values in the dataset.

```python
# Check for missing values in each column
missing_values = df.isnull().sum()
print("Missing values per column:\n", missing_values)
```

Once missing values are detected, they can be handled in different ways. If a column has too many missing values, it might be better to remove it.

```python
df.dropna(inplace=True)  # Removes all rows with missing values
```

If only a few values are missing, we can replace them using statistical methods (Imputation).

```python
df["Math_Score"].fillna(df["Math_Score"].mean(), inplace=True)  # Fill with mean value
df["Extra_Curricular"].fillna("No", inplace=True)  # Fill categorical with most common value
```

For missing values in sequential data (for example, Time-Series Data), we can use forward or backward fill.

```python
df.fillna(method="ffill", inplace=True)  # Forward fill
df.fillna(method="bfill", inplace=True)  # Backward fill
```
#### 4. Outlier Detection
Outliers can be detected using boxplots or Z-score methods.

**Python Code:**
```python
# Boxplot for outlier detection
sns.boxplot(y=df["Math_Score"])
plt.title("Boxplot of Math Scores")
plt.show()
```

### Bivariate Analysis
Bivariate analysis examines the relationship between two variables, often using scatter plots or correlation coefficients.

```python
# Scatter plot between Hours_Studied and Math_Score
sns.scatterplot(x=df["Hours_Studied"], y=df["Math_Score"])
plt.title("Hours Studied vs. Math Score")
plt.show()
```

### Multivariate Analysis
Multivariate analysis involves analyzing more than two variables simultaneously.

```python
# Pairplot for visualizing multiple relationships
sns.pairplot(df[["Math_Score", "Reading_Score", "Writing_Score", "Hours_Studied"]])
plt.show()
```

### Correlational Analysis
Correlation measures how variables move together.

**Python Code:**
```python
# Compute correlation matrix
correlation_matrix = df[["Math_Score", "Reading_Score", "Writing_Score", "Hours_Studied"]].corr()
print(correlation_matrix)

# Heatmap visualization
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
```

## Feature Engineering
Feature Engineering is the process of creating new features from existing ones to improve model performance. Effective feature engineering can significantly enhance the predictive power of machine learning models.

### Creating New Features
For example, creating an "Average Score" feature.

**Python Code:**
```python
df["Average_Score"] = (df["Math_Score"] + df["Reading_Score"] + df["Writing_Score"]) / 3
print(df.head())
```

### Creating Interaction Features
Interaction features are derived from two or more existing features.

**Python Code:**
```python
df["Study_Efficiency"] = df["Math_Score"] / df["Hours_Studied"]
print(df[["Math_Score", "Hours_Studied", "Study_Efficiency"]])
```

### One-Hot Encoding
One-Hot encoding converts categorical features into numerical format for machine learning models.

**Python Code:**
```python
df = pd.get_dummies(df, columns=["Gender", "Extra_Curricular"], drop_first=True)
print(df.head())
```

### Binarizing
Binarizing is an feature engineering technique where continuous numerical data is converted into a binary feature to be used in machine learning models. The below example demonstrates feature engineering by binarizing the "Hours_Studied" feature into a new column "High_Study_Hours". A threshold of 5 hours is used:

- 1 (High Study Hours) → If Hours_Studied is greater than or equal to 5
- 0 (Low Study Hours) → If Hours_Studied is less than 5

```python
# Define the binarization threshold
threshold = 5

# Initialize the Binarizer
binarizer = Binarizer(threshold=threshold)

# Apply binarization to create the new feature
df["High_Study_Hours"] = binarizer.fit_transform(df[["Hours_Studied"]])

# Display the modified dataset
print(df)
```

### Feature Scaling
Feature scaling standardizes numerical features for better model performance.

**Python Code:**
```python
# Initialize the Scaler
scaler = StandardScaler()
# Perform scaling using the mean and standard deviation of each feature
df[["Math_Score", "Reading_Score", "Writing_Score", "Hours_Studied"]] = scaler.fit_transform(
    df[["Math_Score", "Reading_Score", "Writing_Score", "Hours_Studied"]]
)
print(df.head())
```

## **Activity**: Applying Exploratory Data Analysis

### **Objective**
Reinforce understanding of EDA concepts by having participants analyze a small dataset, identify patterns, and suggest improvements for feature engineering.

#### **Scenario**
You are tasked with performing EDA on a dataset of customer information to better understand purchase behavior. The dataset includes:

| CustomerID | Age | Income ($) | Purchased |
|------------|-----|------------|-----------|
| 1          | 25  | 40000      | Yes       |
| 2          | 35  | 50000      | No        |
| 3          | 45  | 60000      | Yes       |
| 4          | 50  | 70000      | No        |
| 5          | 23  | 35000      | Yes       |
| 6          | 40  | 65000      | No        |

#### **Instructions**
Analyze the dataset to answer the following questions:

1. **Univariate Analysis**:  
   - What is the mean and range of the `Age` column?  
   - What does this tell you about the age distribution of the customers?

2. **Bivariate Analysis**:  
   - Analyze the relationship between `Income` and `Purchased`.  
   - Are there patterns indicating how income influences purchase decisions?

3. **Correlation Analysis**:  
   - Calculate or hypothesize the correlation between `Age` and `Income`.  
   - What does a high or low correlation suggest in this context?

4. **Outlier Detection**:  
   - Using the provided IQR method, check if there are any outliers in the `Income` column.  
   - If outliers exist, what impact might they have on model performance?

5. **Feature Engineering**:  
   - Create a new feature, `Income per Year of Age`, for the dataset.  
   - What insights can be derived from this new feature? How could it improve your model?


#### **Expected Outcomes**
- Participants will calculate and interpret key statistics for the dataset.
- Insights about relationships between variables will emerge (e.g., lower-income customers are more likely to purchase).
- Participants will understand the importance of feature engineering and its impact on predictive models.

#### **Discussion Prompts**
- How did EDA help you understand the underlying patterns in the dataset?  
- What challenges did you face when identifying relationships or anomalies?  
- How would you refine the dataset or features further before modeling?

## **Key Takeaways**
- EDA is critical for understanding the data and guiding preprocessing and feature selection.
- Techniques like univariate, bivariate, and multivariate analyses uncover valuable insights.
- Feature engineering can significantly enhance the relevance and impact of the dataset.
