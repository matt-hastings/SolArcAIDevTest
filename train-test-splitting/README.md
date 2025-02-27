<h1>
  <span class="headline">ML Workflow and Best Practices</span>
  <span class="subhead">Train-Test Splitting</span>
</h1>

**Learning objective:** By the end of this lesson, you will be able to perform train-test splitting on the available dataset for building a machine learning model. 

## An Introduction to Train-Test Splitting
Imagine a student preparing for an exam. If they only study from a single book (training set) and are tested with the same book, they may score well. However, if tested on a different book (test set), they might struggle if they haven't understood the concepts properly.

Similarly, machine learning models must be evaluated on **unseen data** to ensure they can **generalize well**. The **train-test split** is a common technique used in machine learning to evaluate the performance of a model. It involves dividing a dataset into two subsets: 

1. **Training Set**: This subset is used to train the model. The model learns patterns and relationships from this data.
2. **Testing Set**: This subset is used to test the model's performance on unseen data. It evaluates how well the model generalizes to new, unseen instances.

## Importance of Train-Test Split
- **Overfitting** occurs when a model learns too much from the training data and fails on new, unseen data.  
- **Underfitting** occurs when a model is too simple and fails to learn meaningful patterns from the data. 
- Without a proper train-test split, the model may **memorize** the training data instead of learning patterns, leading to **overfitting**.
- A **well-balanced train-test split** ensures that the model learns **generalizable patterns** rather than just memorizing training examples. 
- A **well-balanced train-test split** gives confidence in the model’s performance before deploying it in the real world.  


## Demo of Train-Test Splitting
We use the `train_test_split()` function from `sklearn.model_selection` to split data into training and testing sets. The below code demonstrates **train-test splitting on a student performance dataset**.

### Step 1: Import the Required Libraries (_if not done already_)
```python
import pandas as pd
from sklearn.model_selection import train_test_split
```

### Step 2: Load the dataset
```python
# Sample dataset
data = {
    "Student_ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Math_Score": [85, 92, 78, 70, 60, 88, 95, 77, 82, 68],
    "Reading_Score": [78, 88, 74, 65, 58, 85, 91, 72, 79, 66],
    "Writing_Score": [80, 90, 75, 68, 55, 87, 94, 73, 81, 64],
}

df = pd.DataFrame(data)
print(df)
```

### Step 3: Split the dataset into Training and Testing Sets 
```python
# Features and Target Variable
X = df[["Math_Score", "Reading_Score"]]  # Features
y = df["Writing_Score"]  # Target variable

# Perform train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print dataset sizes
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)
```

- `test_size=0.2` → **80% of the data is used for training, and 20% for testing.**  
- `random_state=42` → Ensures **reproducibility** (same split every time).  
- `X_train, X_test, y_train, y_test` → Holds the split data for training and testing.


## Choosing the Right Train-Test Split Ratio

 | **Dataset Size**                 | **Common Train-Test Split Ratio**  | **Why?**                                        |
|----------------------------------|----------------------------------|------------------------------------------------|
| **Small dataset (<1,000 samples)**  | **90% train, 10% test**         | To retain as much training data as possible   |
| **Medium dataset (1,000 - 10,000 samples)** | **80% train, 20% test**  | Standard practice                              |
| **Large dataset (>10,000 samples)** | **70% train, 30% test**         | Sufficient training data available             |
| **Very large dataset (>100,000 samples)** | **60% train, 20% validation, 20% test** | Model tuning and performance evaluation |


### When to Use a Validation Set?
- If hyperparameter tuning is required, a **third set called validation set** is needed.  
- A typical split in this case:  
  - **60% training**  
  - **20% validation** (for model tuning)  
  - **20% testing** (final evaluation)  

**Train-Validation-Test Split:**  
```python
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)  # First split (60% train)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # Second split (20% validation, 20% test)
```

## Independent Practice
- Try running the provided Python code on Jupyter Notebook.
- Experiment with different train-test split ratios and analyze model performance. 
- Explore train-validation-test splits for hyperparameter tuning.

