<h1>
  <span class="headline">ML Workflow and Best Practices</span>
  <span class="subhead">Machine Learning Workflow</span>
</h1>

**Learning Objectives**:  
By the end of this lesson, you will be able to:
- Apply a structured machine learning workflow to solve business problems
- Identify the appropriate steps needed for different types of ML projects
- Evaluate the effectiveness of each workflow stage in a real-world context

## An Introduction to Machine Learning Workflow

The **Machine Learning (ML) workflow** is a systematic approach that transforms raw data into actionable insights through a series of well-defined steps. This structured process ensures that ML models are developed efficiently and deployed effectively to solve real-world problems.

<div class="mermaid">
graph TD
    A[Problem Definition] --> B[Data Collection]
    B --> C[Data Preprocessing]
    C --> D[Exploratory Data Analysis]
    D --> E[Train-Test Splitting]
    E --> F[Model Selection & Training]
    F --> G[Bias-Variance Tradeoff]
    G --> H[Cross-Validation]
    H --> I[Model Monitoring]
    I -.-> A
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style I fill:#bbf,stroke:#333,stroke-width:2px
</div>

### Business Value of a Structured ML Workflow

| Business Need | How the ML Workflow Addresses It |
|---------------|----------------------------------|
| **ROI Justification** | Provides clear documentation of process and outcomes for stakeholders |
| **Resource Optimization** | Streamlines development, reducing redundant work and technical debt |
| **Risk Management** | Systematic validation reduces the chance of model failure in production |
| **Scalability** | Standardized approach allows for consistent deployment across the organization |
| **Regulatory Compliance** | Documented process helps meet requirements for model transparency |

The ML workflow serves as a roadmap for both beginners and professionals, helping them navigate the complexities of data, algorithms, and deployment strategies. The workflow is iterative, meaning that improvements and adjustments are continuously made to achieve the best results.

## Key Steps in the Machine Learning Workflow 

### Step 1: Problem Definition 

Before building a machine learning model, it's crucial to understand the problem you are solving. This step involves defining objectives, understanding business needs, and determining success metrics.

**Business Application Example:** A retail company wants to reduce customer churn by identifying at-risk customers and offering targeted promotions.

#### Key Activities:
- Translate business objectives into ML tasks (classification, regression, clustering)
- Define clear success metrics (accuracy, precision, recall, ROI)
- Identify constraints (data availability, computational resources, time)
- Align stakeholder expectations

**Code Example: Defining Success Metrics**
```python
# Example of how to define and track success metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# After model prediction
def evaluate_churn_model(y_true, y_pred):
    """Evaluate churn prediction model with business-relevant metrics"""
    results = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),  # Minimize false positives (wasted promotions)
        'recall': recall_score(y_true, y_pred),        # Minimize false negatives (missed at-risk customers)
        'f1': f1_score(y_true, y_pred)                 # Balance between precision and recall
    }
    
    # Business impact calculation
    avg_customer_value = 500  # Average annual customer value ($)
    promotion_cost = 50       # Cost of retention promotion ($)
    
    # True positives: Customers we correctly identified as churning and saved
    saved_customers = sum((y_true == 1) & (y_pred == 1))
    # False positives: Customers we incorrectly identified as churning
    unnecessary_promotions = sum((y_true == 0) & (y_pred == 1))
    
    # Calculate ROI
    revenue_saved = saved_customers * avg_customer_value
    total_promotion_cost = (saved_customers + unnecessary_promotions) * promotion_cost
    roi = (revenue_saved - total_promotion_cost) / total_promotion_cost if total_promotion_cost > 0 else 0
    
    results['estimated_roi'] = roi
    return results
```

#### Key Takeaway for Business:
A well-defined problem with clear success metrics ensures that the ML solution directly addresses business needs and provides measurable value.

### Step 2: Data Collection 

Machine learning models require data to learn from. This step focuses on gathering the right type and amount of data from various sources such as databases, APIs, web scraping, or manually recorded observations.

**Business Application Example:** For the customer churn prediction model, we need to collect historical customer data including purchase history, customer service interactions, and demographic information.

#### Key Activities:
- Identify relevant data sources (internal databases, third-party data)
- Ensure data quality and relevance
- Address data privacy and compliance requirements
- Document data collection methodology

**Code Example: Data Collection from Multiple Sources**
```python
# Example of collecting data from multiple sources
import pandas as pd
import sqlite3
from sqlalchemy import create_engine

# Connect to customer database
conn = sqlite3.connect('customer_database.db')
customer_data = pd.read_sql("SELECT * FROM customers WHERE join_date >= '2022-01-01'", conn)

# Connect to transactions database
engine = create_engine('postgresql://username:password@localhost:5432/transactions')
transaction_data = pd.read_sql("SELECT customer_id, purchase_date, amount FROM purchases", engine)

# Load customer service interactions from CSV
service_data = pd.read_csv('customer_service_interactions.csv')

# Merge datasets on customer_id
customer_full_data = customer_data.merge(
    transaction_data.groupby('customer_id').agg({
        'amount': ['sum', 'mean', 'count'],
        'purchase_date': ['max']
    }).reset_index(),
    on='customer_id',
    how='left'
)

customer_full_data = customer_full_data.merge(
    service_data.groupby('customer_id').size().reset_index(name='num_service_interactions'),
    on='customer_id',
    how='left'
)

# Fill missing values for customers with no transactions or service interactions
customer_full_data = customer_full_data.fillna({
    ('amount', 'sum'): 0,
    ('amount', 'mean'): 0,
    ('amount', 'count'): 0,
    'num_service_interactions': 0
})

print(f"Collected data for {len(customer_full_data)} customers with {customer_full_data.columns.size} features")
```

#### Key Takeaway for Business:
Comprehensive data collection that combines multiple relevant sources provides a more complete view of customer behavior, leading to more accurate predictions and better business decisions.

### Step 3: Data Preprocessing 

Raw data is often messy, containing missing values, duplicate entries, or incorrect formats. Data preprocessing cleans and transforms the data to make it suitable for modeling.

**Business Application Example:** Preparing customer data by handling missing values, standardizing formats, and creating relevant features for churn prediction.

#### Key Activities:
- Handle missing values (imputation or removal)
- Remove duplicates and outliers
- Normalize or standardize numerical features
- Encode categorical variables
- Create derived features that capture business knowledge

**Code Example: Data Preprocessing Pipeline**
```python
# Example of a preprocessing pipeline for customer churn data
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
import numpy as np

# Define feature groups
numeric_features = ['age', 'income', 'tenure_months', 'total_spend', 'avg_monthly_spend']
categorical_features = ['gender', 'subscription_type', 'payment_method']
date_features = ['last_purchase_date', 'signup_date']

# Custom transformer to create date-based features
def create_date_features(X):
    # Convert to datetime if not already
    X = pd.DataFrame(X).apply(pd.to_datetime)
    
    # Calculate days since last purchase
    today = pd.Timestamp.today()
    days_since_purchase = (today - X['last_purchase_date']).dt.days
    
    # Calculate customer tenure in days
    tenure_days = (today - X['signup_date']).dt.days
    
    return np.column_stack((days_since_purchase, tenure_days))

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features),
        
        ('date', Pipeline([
            ('date_features', FunctionTransformer(create_date_features))
        ]), date_features)
    ]
)

# Apply preprocessing
X_processed = preprocessor.fit_transform(customer_data)
print(f"Processed data shape: {X_processed.shape}")
```

#### Key Takeaway for Business:
Proper data preprocessing ensures that the model receives high-quality, consistent inputs, which directly impacts prediction accuracy and the reliability of business decisions based on those predictions.

### Step 4: Exploratory Data Analysis (EDA) 

EDA helps in understanding the data distribution, relationships between variables, and patterns that might affect the model's performance.

**Business Application Example:** Analyzing customer behavior patterns to identify potential indicators of churn.

#### Key Activities:
- Visualize data distributions and relationships
- Identify correlations between features
- Detect patterns and anomalies
- Generate hypotheses about factors influencing the target variable

**Code Example: EDA for Churn Analysis**
```python
# Example of exploratory data analysis for churn prediction
import matplotlib.pyplot as plt
import seaborn as sns

# Set up visualization style
sns.set(style="whitegrid")
plt.figure(figsize=(12, 10))

# 1. Compare feature distributions between churned and non-churned customers
def plot_feature_by_churn(feature, title):
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=customer_data, 
        x=feature, 
        hue="churned",
        multiple="dodge", 
        palette=["green", "red"],
        bins=20
    )
    plt.title(f"{title} Distribution by Churn Status")
    plt.xlabel(title)
    plt.ylabel("Count")
    plt.legend(["Retained", "Churned"])
    plt.tight_layout()
    plt.show()

# Plot key features
plot_feature_by_churn("tenure_months", "Customer Tenure (Months)")
plot_feature_by_churn("avg_monthly_spend", "Average Monthly Spend")
plot_feature_by_churn("num_service_interactions", "Number of Service Interactions")

# 2. Correlation heatmap
plt.figure(figsize=(12, 10))
correlation = customer_data.select_dtypes(include=['number']).corr()
mask = np.triu(correlation)
sns.heatmap(correlation, annot=True, mask=mask, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()

# 3. Churn rate by categorical variables
def plot_churn_by_category(feature, title):
    plt.figure(figsize=(10, 6))
    churn_rate = customer_data.groupby(feature)['churned'].mean().sort_values(ascending=False)
    
    ax = sns.barplot(x=churn_rate.index, y=churn_rate.values, palette="viridis")
    plt.title(f"Churn Rate by {title}")
    plt.xlabel(title)
    plt.ylabel("Churn Rate")
    plt.xticks(rotation=45)
    
    # Add percentage labels on bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{p.get_height():.1%}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'bottom')
    
    plt.tight_layout()
    plt.show()

# Plot churn rate by categories
plot_churn_by_category("subscription_type", "Subscription Type")
plot_churn_by_category("payment_method", "Payment Method")
```

#### Key Takeaway for Business:
EDA reveals actionable insights about customer behavior patterns that can inform both the model development and immediate business strategies, such as identifying high-risk customer segments or problematic service areas.

### Step 5: Train-Test Splitting 

To evaluate the performance of a machine learning model, it should be tested on unseen data. The dataset is split into training and testing sets to assess generalization.

**Business Application Example:** Ensuring the churn prediction model works on new customers, not just those it was trained on.

#### Key Activities:
- Split data into training and testing sets (typically 70-30 or 80-20)
- Ensure representative distribution in both sets
- Consider time-based splitting for time-series data
- Maintain class balance in classification problems

**Code Example: Strategic Train-Test Split**
```python
# Example of train-test splitting with consideration for business context
from sklearn.model_selection import train_test_split

# For time-sensitive problems like churn, consider time-based splitting
customer_data = customer_data.sort_values('last_activity_date')

# Option 1: Simple random split (for non-time-sensitive problems)
X = customer_data.drop('churned', axis=1)
y = customer_data['churned']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Option 2: Time-based split (for forecasting future churn)
# Use the most recent 20% of data as the test set
split_idx = int(len(customer_data) * 0.8)
train_data = customer_data.iloc[:split_idx]
test_data = customer_data.iloc[split_idx:]

X_train = train_data.drop('churned', axis=1)
y_train = train_data['churned']
X_test = test_data.drop('churned', axis=1)
y_test = test_data['churned']

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Churn rate in training set: {y_train.mean():.2%}")
print(f"Churn rate in test set: {y_test.mean():.2%}")
```

#### Key Takeaway for Business:
Proper train-test splitting ensures that model performance metrics reflect how the model will perform on new, unseen customers, providing realistic expectations for business impact.

### Step 6: Model Selection & Training 

After splitting the data, the next step is to choose an appropriate machine learning model based on the problem type and train it using the training dataset.

**Business Application Example:** Selecting and training the most appropriate model for customer churn prediction.

#### Key Activities:
- Select models appropriate for the problem type
- Train models with different algorithms and hyperparameters
- Compare model performance on validation data
- Consider model interpretability requirements for business stakeholders

**Code Example: Model Selection and Training**
```python
# Example of training multiple models for churn prediction
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import pandas as pd

# Define models to evaluate
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced'),
    'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100)
}

# Function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_prob)
    }
    
    # Calculate cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    results['cv_f1_mean'] = cv_scores.mean()
    results['cv_f1_std'] = cv_scores.std()
    
    return results

# Evaluate all models
model_results = {}
for name, model in models.items():
    print(f"Training {name}...")
    model_results[name] = evaluate_model(model, X_train, X_test, y_train, y_test)

# Display results as a DataFrame
results_df = pd.DataFrame(model_results).T
results_df = results_df.sort_values('f1', ascending=False)
print("\nModel Performance Comparison:")
print(results_df.round(3))

# Select best model based on F1 score (balance of precision and recall)
best_model_name = results_df.index[0]
best_model = models[best_model_name]
print(f"\nBest model: {best_model_name} with F1 score: {results_df.loc[best_model_name, 'f1']:.3f}")
```

#### Key Takeaway for Business:
Selecting the right model involves balancing accuracy, interpretability, and business constraints. For churn prediction, a model with high recall ensures you don't miss at-risk customers, while maintaining reasonable precision to avoid wasting resources on false positives.

### Step 7: Bias-Variance Tradeoff  

A model should generalize well to unseen data. If a model is too simple (high bias), it underfits; if too complex (high variance), it overfits.

**Business Application Example:** Finding the right model complexity to accurately predict customer churn without being misled by noise in historical data.

#### Key Activities:
- Analyze learning curves to detect underfitting or overfitting
- Adjust model complexity through regularization or feature selection
- Balance model performance on training and validation data
- Consider the cost of errors in the business context

**Code Example: Analyzing and Addressing Bias-Variance Tradeoff**
```python
# Example of analyzing and addressing bias-variance tradeoff
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(estimator, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='f1'
    )
    
    # Calculate mean and std for training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    
    # Calculate mean and std for test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training Examples")
    plt.ylabel("F1 Score")
    plt.grid()
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()
    
    # Analyze gap between training and test performance
    gap = train_mean[-1] - test_mean[-1]
    if gap > 0.3:
        return "High Variance (Overfitting)"
    elif test_mean[-1] < 0.6:
        return "High Bias (Underfitting)"
    else:
        return "Good Balance"

# Analyze learning curves for different models
for name, model in models.items():
    diagnosis = plot_learning_curve(model, X_train, y_train, f"Learning Curve: {name}")
    print(f"{name}: {diagnosis}")

# Example of addressing high variance (if detected)
if "High Variance" in diagnosis:
    print("\nAddressing overfitting in Random Forest model...")
    # Increase regularization
    rf_tuned = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,           # Limit tree depth
        min_samples_split=10,  # Require more samples to split
        min_samples_leaf=4,    # Require more samples in leaves
        class_weight='balanced'
    )
    
    # Check if tuning helped
    rf_tuned.fit(X_train, y_train)
    y_pred_tuned = rf_tuned.predict(X_test)
    print(f"Original F1: {f1_score(y_test, models['Random Forest'].predict(X_test)):.3f}")
    print(f"Tuned F1: {f1_score(y_test, y_pred_tuned):.3f}")
```

#### Key Takeaway for Business:
Finding the right balance in model complexity ensures that your churn predictions capture real patterns in customer behavior without being misled by random fluctuations, leading to more reliable business decisions and resource allocation.

### Step 8: Cross-Validation 

Before deploying a model, it's crucial to evaluate its performance using metrics like accuracy, precision, recall, RMSE, etc. Cross-validation helps assess how well the model performs on different subsets of the data.

**Business Application Example:** Ensuring the churn prediction model performs consistently across different customer segments and time periods.

#### Key Activities:
- Implement k-fold cross-validation
- Evaluate model stability across different data subsets
- Select appropriate evaluation metrics aligned with business goals
- Consider stratified or time-series cross-validation when appropriate

**Code Example: Strategic Cross-Validation**
```python
# Example of cross-validation for churn prediction
from sklearn.model_selection import cross_validate, StratifiedKFold, TimeSeriesSplit
import pandas as pd

# Define scoring metrics relevant to business goals
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'auc': 'roc_auc'
}

# Option 1: Stratified K-Fold (ensures class balance in each fold)
cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Option 2: Time Series Split (for temporal data)
cv_timeseries = TimeSeriesSplit(n_splits=5)

# Choose the appropriate cross-validation strategy
cv = cv_stratified  # or cv_timeseries if using temporal data

# Perform cross-validation with multiple metrics
best_model = models[best_model_name]
cv_results = cross_validate(
    best_model, X_train, y_train, 
    cv=cv, 
    scoring=scoring,
    return_train_score=True
)

# Create a summary of cross-validation results
cv_summary = pd.DataFrame({
    'Metric': list(scoring.keys()),
    'Mean Train Score': [cv_results[f'train_{metric}'].mean() for metric in scoring.keys()],
    'Train Std Dev': [cv_results[f'train_{metric}'].std() for metric in scoring.keys()],
    'Mean Test Score': [cv_results[f'test_{metric}'].mean() for metric in scoring.keys()],
    'Test Std Dev': [cv_results[f'test_{metric}'].std() for metric in scoring.keys()]
})

print("Cross-Validation Results:")
print(cv_summary.round(3))

# Check for consistency across folds
consistency_check = all(cv_results['test_f1'].std() < 0.05 for metric in scoring.keys())
print(f"\nModel performance consistency: {'Good' if consistency_check else 'Needs improvement'}")

# Business interpretation
print("\nBusiness Interpretation:")
recall = cv_results['test_recall'].mean()
precision = cv_results['test_precision'].mean()
print(f"On average, the model identifies {recall:.1%} of customers who will churn")
print(f"{precision:.1%} of customers flagged as 'at risk' will actually churn")

# Calculate potential business impact
avg_customer_value = 500  # Average annual value of a customer
retention_cost = 50       # Cost of retention offer
total_customers = 10000   # Total customer base
annual_churn_rate = 0.15  # Annual churn rate without intervention

# Estimated customers saved
potential_churners = total_customers * annual_churn_rate
identified_churners = potential_churners * recall
true_positives = identified_churners * precision
saved_customers = true_positives * 0.3  # Assume 30% of identified churners can be saved

# Financial impact
revenue_saved = saved_customers * avg_customer_value
retention_campaign_cost = identified_churners * retention_cost
net_impact = revenue_saved - retention_campaign_cost

print(f"\nEstimated annual financial impact:")
print(f"Revenue saved: ${revenue_saved:,.2f}")
print(f"Retention campaign cost: ${retention_campaign_cost:,.2f}")
print(f"Net impact: ${net_impact:,.2f}")
```

#### Key Takeaway for Business:
Cross-validation provides confidence in your model's performance across different customer segments and time periods, ensuring that your churn prediction strategy will work consistently across your entire customer base.

### Step 9: Model Monitoring

After deployment, continuous monitoring is essential to ensure the model maintains its performance over time and detect any degradation or drift.

**Business Application Example:** Maintaining the effectiveness of the churn prediction model as customer behaviors and market conditions change over time.

#### Key Activities:
- Monitor for model drift (data drift, concept drift, label drift)
- Track performance metrics over time
- Implement automated alerts for performance degradation
- Schedule regular model retraining when necessary

**Code Example: Setting Up Model Monitoring**
```python
# Example of a simple model monitoring system
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Function to simulate monitoring over time
def monitor_model_performance(model, monitoring_period_days=90):
    # Create a dataframe to store monitoring results
    monitoring_results = pd.DataFrame(columns=[
        'date', 'precision', 'recall', 'data_drift_score', 'concept_drift_score'
    ])
    
    # Simulate daily performance checks
    start_date = datetime.now() - timedelta(days=monitoring_period_days)
    
    for day in range(monitoring_period_days):
        current_date = start_date + timedelta(days=day)
        
        # Simulate getting new data for that day
        # In a real system, this would be actual new data
        X_new, y_true = get_new_data_for_date(current_date)  # Placeholder function
        
        # Make predictions
        y_pred = model.predict(X_new)
        
        # Calculate performance metrics
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        
        # Calculate drift scores (simplified examples)
        # In a real system, use proper statistical tests
        data_drift_score = calculate_data_drift(X_train, X_new)  # Placeholder function
        concept_drift_score = calculate_concept_drift(precision, recall)  # Placeholder function
        
        # Add to monitoring results
        monitoring_results = monitoring_results.append({
            'date': current_date,
            'precision': precision,
            'recall': recall,
            'data_drift_score': data_drift_score,
            'concept_drift_score': concept_drift_score
        }, ignore_index=True)
        
        # Check for alerts
        if data_drift_score > 0.3:
            print(f"ALERT: High data drift detected on {current_date.strftime('%Y-%m-%d')}")
            
        if concept_drift_score > 0.3:
            print(f"ALERT: High concept drift detected on {current_date.strftime('%Y-%m-%d')}")
            
        # Check if retraining is needed
        if day % 30 == 0 and (data_drift_score > 0.2 or concept_drift_score > 0.2):
            print(f"Retraining recommended as of {current_date.strftime('%Y-%m-%d')}")
    
    # Visualize monitoring results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(monitoring_results['date'], monitoring_results['precision'], label='Precision')
    plt.plot(monitoring_results['date'], monitoring_results['recall'], label='Recall')
    plt.title('Model Performance Over Time')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(monitoring_results['date'], monitoring_results['data_drift_score'])
    plt.title('Data Drift Score')
    plt.axhline(y=0.2, color='r', linestyle='--', label='Alert Threshold')
    
    plt.subplot(3, 1, 3)
    plt.plot(monitoring_results['date'], monitoring_results['concept_drift_score'])
    plt.title('Concept Drift Score')
    plt.axhline(y=0.2, color='r', linestyle='--', label='Alert Threshold')
    
    plt.tight_layout()
    plt.show()
    
    return monitoring_results

# Placeholder functions for the example
def get_new_data_for_date(date):
    # In a real system, this would retrieve actual new data
    # Here we're just simulating data with gradual drift
    X = X_test.copy()
    y = y_test.copy()
    
    # Simulate drift based on date
    days_passed = (date - (datetime.now() - timedelta(days=90))).days
    drift_factor = days_passed / 90  # Increases over time
    
    # Add noise proportional to drift_factor
    X = X + np.random.normal(0, drift_factor * 0.5, X.shape)
    
    # For concept drift, we could modify the relationship between X and y
    # This is a simplified example
    
    return X, y

def calculate_data_drift(X_original, X_new):
    # In a real system, use proper statistical tests
    # This is a simplified example using mean difference
    return np.mean(np.abs(X_original.mean(axis=0) - X_new.mean(axis=0)))

def calculate_concept_drift(current_precision, current_recall):
    # In a real system, use proper statistical tests
    # This is a simplified example using the difference in precision and recall
    return abs(current_precision - 0.85) + abs(current_recall - 0.80)  # Assuming baseline values
```

#### Key Takeaway for Business:
Regular model monitoring ensures that your churn prediction model remains effective as customer behaviors evolve, market conditions change, and new products are introduced. Early detection of performance degradation allows for timely intervention, maintaining the ROI of your customer retention efforts.

## Discussion Activity: Understanding the Importance of the ML Workflow
### Objective:
Reflect on the key components and significance of the ML workflow in real-world applications.
### Discussion Prompts:
  - Why is having a well-defined workflow important for developing machine learning models? 
  - Can you think of examples where a lack of structure in the workflow might lead to inefficiencies or poor outcomes?
