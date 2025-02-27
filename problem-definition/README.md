<h1>
  <span class="headline">ML Workflow and Best Practices</span>
  <span class="subhead">Problem Definition</span>
</h1>

**Learning Objectives:**  
By the end of this lesson, you will be able to:
- Formulate business problems as machine learning tasks with clear success metrics
- Translate business requirements into appropriate ML problem types (classification, regression, clustering)
- Design evaluation frameworks that align ML model performance with business objectives
- Identify constraints and requirements that impact ML solution design

## The Critical First Step: Defining the Problem

Problem definition is the foundation of any successful machine learning project. A well-defined problem aligns technical implementation with business goals, ensuring that your ML solution delivers real value. This step determines everything that follows in the ML workflow.

<div class="mermaid">
graph TD
    A[Business Problem] --> B[ML Problem Definition]
    B --> C[Data Requirements]
    B --> D[Success Metrics]
    B --> E[Constraints]
    C --> F[Data Collection Strategy]
    D --> G[Evaluation Framework]
    E --> H[Solution Design]
    F --> I[ML Implementation]
    G --> I
    H --> I
    I --> J[Business Value]
</div>

### Why Problem Definition Matters

| Poor Problem Definition | Strong Problem Definition |
|-------------------------|---------------------------|
| Misaligned with business goals | Directly addresses business needs |
| Vague success criteria | Clear, measurable success metrics |
| Undefined constraints | Well-understood limitations and requirements |
| Scope creep | Focused implementation |
| Wasted resources | Efficient resource allocation |
| Difficult to evaluate success | Clear evaluation framework |

## Step 1: Understand the Business Context

Before diving into technical implementation, you must thoroughly understand the business problem you're trying to solve. This requires collaboration with stakeholders and domain experts.

<div class="mermaid">
graph LR
    A[Business Stakeholders] --> B[Business Problem]
    C[Domain Experts] --> B
    D[Data Scientists] --> B
    B --> E[ML Problem Definition]
</div>

### Key Questions to Ask:

1. **What business goal are we trying to achieve?**
   - Increase revenue? Reduce costs? Improve customer experience?
   - What is the current process, and why does it need improvement?

2. **Who will use the model's predictions?**
   - How will they incorporate the predictions into their workflow?
   - What level of interpretability do they need?

3. **What decisions will be made based on the model's output?**
   - What actions will be taken based on predictions?
   - What is the cost of incorrect predictions?

4. **What data is available or could be collected?**
   - What historical data exists that relates to this problem?
   - Are there any data collection or privacy constraints?

### Business Example: Customer Churn Reduction

**Initial Business Request:** "We need to reduce customer churn."

**Improved Problem Definition After Stakeholder Discussions:**
- **Business Goal:** Reduce monthly customer churn rate from 5% to 3% within 6 months
- **Users:** Customer retention team who will contact at-risk customers
- **Decisions:** Allocate retention resources (special offers, personalized outreach) to customers most likely to churn
- **Available Data:** Customer demographics, purchase history, service usage, support interactions
- **Constraints:** Retention budget allows contacting only 10% of customers per month

## Step 2: Translate to a Machine Learning Problem Type

Once you understand the business context, translate it into a specific type of machine learning problem.

<div class="mermaid">
graph TD
    A[Business Problem] --> B{Problem Type?}
    B -->|Predict a category| C[Classification]
    B -->|Predict a number| D[Regression]
    B -->|Find patterns/groups| E[Clustering]
    B -->|Recommend items| F[Recommendation]
    B -->|Detect anomalies| G[Anomaly Detection]
    C --> H[Binary or Multi-class?]
    D --> I[Linear or Non-linear?]
    E --> J[Number of clusters known?]
</div>

### Common ML Problem Types:

| Problem Type | When to Use | Business Examples |
|--------------|-------------|-------------------|
| **Classification** | Predicting categories or classes | Customer churn prediction, Fraud detection, Email spam filtering |
| **Regression** | Predicting numerical values | Sales forecasting, Price optimization, Demand prediction |
| **Clustering** | Finding natural groupings | Customer segmentation, Product categorization, Market analysis |
| **Recommendation** | Suggesting relevant items | Product recommendations, Content personalization, Next best action |
| **Anomaly Detection** | Identifying unusual patterns | Fraud detection, Equipment failure prediction, Network security |

### Code Example: Defining a Classification Problem

```python
# Example: Defining a customer churn prediction problem
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load customer data
customer_data = pd.read_csv('customer_data.csv')

# Examine the data to understand what we're working with
print(f"Dataset shape: {customer_data.shape}")
print(f"Columns: {customer_data.columns.tolist()}")

# Define the target variable (what we want to predict)
# In this case, 'churned' is 1 if the customer left, 0 if they stayed
target = 'churned'

# Check class distribution (how many churned vs. non-churned customers)
churn_distribution = customer_data[target].value_counts(normalize=True)
print(f"Churn distribution:\n{churn_distribution}")

# Define features (predictors) - everything except the target
features = [col for col in customer_data.columns if col != target]

# Split data into features (X) and target (y)
X = customer_data[features]
y = customer_data[target]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Define the problem type and approach
problem_definition = {
    'business_objective': 'Reduce customer churn rate from 5% to 3%',
    'ml_problem_type': 'Binary Classification',
    'target_variable': 'churned (1=Yes, 0=No)',
    'features': features,
    'success_metrics': ['Recall (catch at least 80% of churners)', 
                       'Precision (at least 60% of predicted churners actually churn)'],
    'constraints': ['Model must be interpretable for business users',
                   'Predictions needed 30 days in advance',
                   'Can only contact 10% of customers per month']
}

# Document the problem definition
for key, value in problem_definition.items():
    print(f"{key.replace('_', ' ').title()}: {value}")
```

### Business Example: Translating Churn Reduction to ML

**Business Problem:** Reduce customer churn rate from 5% to 3%

**ML Problem Type:** Binary classification (Will a customer churn? Yes/No)

**Target Variable:** Customer churn status (1 = churned, 0 = retained)

**Features:** Customer demographics, purchase history, service usage, support interactions

**Success Metrics:** 
- Recall: Identify at least 80% of customers who will churn
- Precision: At least 60% of customers flagged as "likely to churn" actually churn

## Step 3: Define Clear Success Metrics

Success metrics translate business goals into measurable technical criteria. They should directly connect model performance to business impact.

<div class="mermaid">
graph TD
    A[Business Goal] --> B[Technical Metrics]
    B --> C[Evaluation Framework]
    C --> D[Model Selection]
    C --> E[Hyperparameter Tuning]
    C --> F[Business Impact Assessment]
</div>

### Choosing the Right Metrics:

| Problem Type | Common Metrics | Business Considerations |
|--------------|----------------|-------------------------|
| **Classification** | Accuracy, Precision, Recall, F1-score, AUC-ROC | Cost of false positives vs. false negatives |
| **Regression** | MAE, MSE, RMSE, R² | Acceptable error range, impact of outliers |
| **Clustering** | Silhouette score, Davies-Bouldin index | Business interpretability of clusters |
| **Recommendation** | Precision@k, Recall@k, NDCG | User engagement, revenue impact |
| **Anomaly Detection** | Precision, Recall, F1-score | Cost of missed anomalies vs. false alarms |

### Code Example: Defining Success Metrics for Churn Prediction

```python
# Example: Defining and evaluating success metrics for churn prediction
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score

# Continuing from previous example with X_train, y_train already defined

# Define a simple baseline model
model = RandomForestClassifier(random_state=42)

# Define the metrics we care about based on business requirements
def evaluate_churn_model(model, X, y, threshold=0.5):
    """
    Evaluate a churn prediction model with business-relevant metrics
    
    Parameters:
    - model: The trained model
    - X: Features
    - y: True labels
    - threshold: Probability threshold for classifying as churn (default: 0.5)
    
    Returns:
    - Dictionary of metrics and business impact
    """
    # Get predictions and probabilities
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    # Calculate standard metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),  # Of those predicted to churn, how many actually did
        'recall': recall_score(y, y_pred),        # Of those who churned, how many did we catch
        'f1': f1_score(y, y_pred),                # Harmonic mean of precision and recall
        'auc': roc_auc_score(y, y_prob)           # Area under ROC curve
    }
    
    # Calculate business impact
    total_customers = len(y)
    actual_churners = sum(y)
    predicted_churners = sum(y_pred)
    correctly_predicted_churners = sum((y == 1) & (y_pred == 1))
    
    # Business metrics
    avg_customer_value = 500  # Average annual value of a customer
    retention_cost = 50       # Cost per customer for retention campaign
    retention_success_rate = 0.3  # 30% of correctly identified churners can be saved
    
    # Calculate financial impact
    saved_customers = correctly_predicted_churners * retention_success_rate
    retention_campaign_cost = predicted_churners * retention_cost
    saved_revenue = saved_customers * avg_customer_value
    net_benefit = saved_revenue - retention_campaign_cost
    
    # Add business metrics to results
    metrics['identified_churners_pct'] = correctly_predicted_churners / actual_churners
    metrics['retention_campaign_cost'] = retention_campaign_cost
    metrics['saved_revenue'] = saved_revenue
    metrics['net_benefit'] = net_benefit
    metrics['roi'] = (saved_revenue - retention_campaign_cost) / retention_campaign_cost if retention_campaign_cost > 0 else 0
    
    return metrics

# Evaluate with cross-validation to get a reliable estimate
from sklearn.model_selection import cross_validate

# Define custom scoring functions for cross_validate
from sklearn.metrics import make_scorer

# Custom scorer for business impact
def business_impact_scorer(y_true, y_pred):
    # Simplified version of the business impact calculation
    correctly_predicted_churners = sum((y_true == 1) & (y_pred == 1))
    predicted_churners = sum(y_pred)
    
    avg_customer_value = 500
    retention_cost = 50
    retention_success_rate = 0.3
    
    saved_customers = correctly_predicted_churners * retention_success_rate
    retention_campaign_cost = predicted_churners * retention_cost
    saved_revenue = saved_customers * avg_customer_value
    net_benefit = saved_revenue - retention_campaign_cost
    
    return net_benefit

# Create a custom scorer
business_scorer = make_scorer(business_impact_scorer, greater_is_better=True)

# Define scoring metrics for cross-validation
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'auc': 'roc_auc',
    'business_impact': business_scorer
}

# Perform cross-validation with multiple metrics
cv_results = cross_validate(
    model, X_train, y_train, 
    cv=5, 
    scoring=scoring,
    return_train_score=True
)

# Print average results across folds
print("Cross-Validation Results:")
for metric in scoring.keys():
    mean_score = cv_results[f'test_{metric}'].mean()
    std_score = cv_results[f'test_{metric}'].std()
    print(f"{metric}: {mean_score:.4f} (±{std_score:.4f})")

# Check if we meet our business requirements
meets_recall_requirement = cv_results['test_recall'].mean() >= 0.8
meets_precision_requirement = cv_results['test_precision'].mean() >= 0.6

print(f"\nMeets recall requirement (≥80%): {meets_recall_requirement}")
print(f"Meets precision requirement (≥60%): {meets_precision_requirement}")

if meets_recall_requirement and meets_precision_requirement:
    print("The model meets all business requirements!")
else:
    print("The model does not meet all business requirements. Further tuning needed.")
```

### Business Example: Success Metrics for Churn Prediction

**Technical Metrics:**
- **Recall ≥ 80%**: We need to identify at least 80% of customers who will churn
- **Precision ≥ 60%**: At least 60% of customers we flag as "likely to churn" should actually churn

**Business Impact Metrics:**
- **Retention Campaign ROI**: (Value of saved customers - Cost of retention campaign) / Cost of retention campaign
- **Net Churn Reduction**: Percentage point reduction in overall churn rate
- **Customer Lifetime Value Preserved**: Total value of customers saved from churning

## Step 4: Identify Constraints and Requirements

Constraints and requirements shape the solution design and implementation. They must be identified early to avoid wasted effort.

<div class="mermaid">
graph TD
    A[Constraints & Requirements] --> B[Technical Constraints]
    A --> C[Business Constraints]
    A --> D[Ethical & Legal Requirements]
    B --> E[Solution Design]
    C --> E
    D --> E
</div>

### Common Constraints to Consider:

| Constraint Type | Examples | Impact on ML Solution |
|-----------------|----------|------------------------|
| **Technical** | Computation resources, Deployment environment, Latency requirements | Model complexity, Algorithm selection, Feature engineering approach |
| **Data** | Data availability, Data quality, Privacy restrictions | Feature selection, Data collection strategy, Model accuracy |
| **Business** | Budget, Timeline, Expertise available | Project scope, Implementation approach, Tool selection |
| **Ethical & Legal** | Fairness requirements, Regulatory compliance, Explainability needs | Algorithm selection, Feature inclusion, Documentation requirements |

### Code Example: Documenting Constraints and Requirements

```python
# Example: Documenting constraints and requirements for a churn prediction project

# Define project constraints and requirements
project_constraints = {
    'technical': {
        'deployment_environment': 'On-premise server with 16GB RAM',
        'prediction_latency': 'Batch predictions, run nightly',
        'integration': 'Must integrate with existing CRM system via API'
    },
    'data': {
        'available_history': '3 years of customer data',
        'data_refresh_rate': 'Daily updates',
        'privacy_restrictions': 'Cannot use personally identifiable information'
    },
    'business': {
        'timeline': 'Model must be deployed within 2 months',
        'budget': 'Limited to existing team resources',
        'interpretability': 'Business users must understand key factors driving churn'
    },
    'ethical_legal': {
        'fairness': 'Model must not discriminate based on protected attributes',
        'transparency': 'Decision process must be documentable for compliance',
        'data_usage': 'Customer data usage must comply with privacy policy'
    }
}

# Function to check if a model meets the constraints
def check_model_against_constraints(model, X, constraints):
    """
    Evaluate if a model meets the defined constraints
    
    Parameters:
    - model: The trained model
    - X: Feature data
    - constraints: Dictionary of project constraints
    
    Returns:
    - Dictionary of constraint checks
    """
    import time
    import sys
    from sklearn.inspection import permutation_importance
    
    constraint_checks = {}
    
    # Check technical constraints
    # Memory usage
    memory_before = sys.getsizeof(model)
    
    # Latency
    start_time = time.time()
    model.predict(X)
    prediction_time = time.time() - start_time
    avg_prediction_time_ms = (prediction_time / len(X)) * 1000
    
    constraint_checks['technical'] = {
        'model_size_kb': memory_before / 1024,
        'avg_prediction_time_ms': avg_prediction_time_ms,
        'meets_latency_requirement': avg_prediction_time_ms < 100  # Example threshold
    }
    
    # Check interpretability requirements
    if hasattr(model, 'feature_importances_'):
        # Direct feature importances (e.g., for tree-based models)
        feature_importance = model.feature_importances_
    else:
        # Permutation importance for black-box models
        result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        feature_importance = result.importances_mean
    
    # Get top 5 features
    feature_names = X.columns if hasattr(X, 'columns') else [f"Feature {i}" for i in range(X.shape[1])]
    top_features = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)[:5]
    
    constraint_checks['business'] = {
        'interpretable': isinstance(model, (RandomForestClassifier, 
                                           'DecisionTreeClassifier', 
                                           'LogisticRegression')),
        'top_features': top_features
    }
    
    return constraint_checks

# Example usage (assuming model is already trained)
# constraint_results = check_model_against_constraints(model, X_test, project_constraints)
# print(constraint_results)
```

### Business Example: Constraints for Churn Prediction

**Technical Constraints:**
- Model must run on existing infrastructure (no cloud deployment)
- Predictions must be available daily for the retention team
- Must integrate with the company's CRM system

**Data Constraints:**
- Only 3 years of historical customer data available
- Cannot use certain sensitive customer information due to privacy policies
- Data quality issues with older records

**Business Constraints:**
- Retention team can only contact 10% of customers per month
- Model must be deployed within 2 months
- Solution must be maintainable by the existing data science team

**Ethical & Legal Requirements:**
- Model must not discriminate based on protected attributes
- Decision process must be explainable to customers
- Data usage must comply with privacy regulations

## Step 5: Document the Problem Definition

A well-documented problem definition serves as a reference throughout the project and helps align all stakeholders.

### Problem Definition Template

```python
# Example: Comprehensive problem definition document

def create_problem_definition_document(
    business_context,
    ml_problem_type,
    success_metrics,
    constraints,
    stakeholders
):
    """
    Create a comprehensive problem definition document
    
    Parameters:
    - business_context: Dict with business goal, current process, etc.
    - ml_problem_type: Dict with problem type, target variable, etc.
    - success_metrics: Dict with technical and business metrics
    - constraints: Dict with technical, data, business constraints
    - stakeholders: Dict with key stakeholders and their requirements
    
    Returns:
    - Formatted problem definition document
    """
    import json
    from datetime import datetime
    
    # Combine all information
    problem_definition = {
        'project_name': business_context.get('project_name', 'Unnamed ML Project'),
        'created_date': datetime.now().strftime('%Y-%m-%d'),
        'business_context': business_context,
        'ml_problem_type': ml_problem_type,
        'success_metrics': success_metrics,
        'constraints': constraints,
        'stakeholders': stakeholders,
        'approval_status': 'Draft'
    }
    
    # Create formatted document
    document = f"""
    # Machine Learning Problem Definition
    
    ## Project: {problem_definition['project_name']}
    Created: {problem_definition['created_date']}
    
    ## 1. Business Context
    
    **Business Goal:** {business_context['goal']}
    
    **Current Process:** {business_context['current_process']}
    
    **Expected Impact:** {business_context['expected_impact']}
    
    ## 2. Machine Learning Problem
    
    **Problem Type:** {ml_problem_type['type']}
    
    **Target Variable:** {ml_problem_type['target_variable']}
    
    **Features:** {', '.join(ml_problem_type.get('features', ['To be determined']))}
    
    ## 3. Success Metrics
    
    **Technical Metrics:**
    """
    
    for metric, details in success_metrics['technical'].items():
        document += f"- {metric}: {details}\n"
    
    document += """
    **Business Metrics:**
    """
    
    for metric, details in success_metrics['business'].items():
        document += f"- {metric}: {details}\n"
    
    document += """
    ## 4. Constraints and Requirements
    
    **Technical Constraints:**
    """
    
    for constraint, details in constraints['technical'].items():
        document += f"- {constraint}: {details}\n"
    
    document += """
    **Data Constraints:**
    """
    
    for constraint, details in constraints['data'].items():
        document += f"- {constraint}: {details}\n"
    
    document += """
    **Business Constraints:**
    """
    
    for constraint, details in constraints['business'].items():
        document += f"- {constraint}: {details}\n"
    
    document += """
    **Ethical & Legal Requirements:**
    """
    
    for requirement, details in constraints['ethical_legal'].items():
        document += f"- {requirement}: {details}\n"
    
    document += """
    ## 5. Key Stakeholders
    """
    
    for role, details in stakeholders.items():
        document += f"- **{role}**: {details['name']} - {details['requirements']}\n"
    
    document += """
    ## 6. Approval
    
    Status: {approval_status}
    
    Approvers:
    """
    
    for approver in stakeholders.get('approvers', []):
        document += f"- [ ] {approver}\n"
    
    # Also save as JSON for programmatic access
    with open(f"{problem_definition['project_name'].replace(' ', '_').lower()}_problem_definition.json", 'w') as f:
        json.dump(problem_definition, f, indent=2)
    
    return document

# Example usage
business_context = {
    'project_name': 'Customer Churn Prediction',
    'goal': 'Reduce monthly customer churn rate from 5% to 3% within 6 months',
    'current_process': 'Reactive approach - retention team contacts customers after they indicate intent to cancel',
    'expected_impact': 'Proactive retention efforts will save approximately $2M in annual revenue'
}

ml_problem_type = {
    'type': 'Binary Classification',
    'target_variable': 'churned (1=Yes, 0=No)',
    'features': ['customer_tenure', 'monthly_charges', 'total_charges', 'contract_type', 
                'payment_method', 'internet_service', 'tech_support', 'online_security']
}

success_metrics = {
    'technical': {
        'recall': 'At least 80% (catch most potential churners)',
        'precision': 'At least 60% (minimize wasted retention efforts)',
        'auc': 'At least 0.85 (good overall discrimination)'
    },
    'business': {
        'churn_reduction': 'Decrease churn rate from 5% to 3%',
        'roi': 'At least 300% return on retention campaign investment',
        'net_revenue_impact': 'Increase net revenue by at least $1.5M annually'
    }
}

constraints = {
    'technical': {
        'deployment_environment': 'On-premise server with 16GB RAM',
        'prediction_latency': 'Batch predictions, run nightly',
        'integration': 'Must integrate with existing CRM system via API'
    },
    'data': {
        'available_history': '3 years of customer data',
        'data_refresh_rate': 'Daily updates',
        'privacy_restrictions': 'Cannot use personally identifiable information'
    },
    'business': {
        'timeline': 'Model must be deployed within 2 months',
        'budget': 'Limited to existing team resources',
        'interpretability': 'Business users must understand key factors driving churn'
    },
    'ethical_legal': {
        'fairness': 'Model must not discriminate based on protected attributes',
        'transparency': 'Decision process must be documentable for compliance',
        'data_usage': 'Customer data usage must comply with privacy policy'
    }
}

stakeholders = {
    'business_owner': {
        'name': 'Sarah Johnson, VP of Customer Success',
        'requirements': 'Needs clear ROI and implementation plan'
    },
    'end_users': {
        'name': 'Customer Retention Team',
        'requirements': 'Need actionable predictions and explanation of risk factors'
    },
    'technical_owner': {
        'name': 'Michael Chen, Data Science Manager',
        'requirements': 'Needs maintainable solution with documented methodology'
    },
    'approvers': ['Sarah Johnson', 'Michael Chen', 'Alex Rodriguez (CTO)']
}

# Generate the problem definition document
problem_definition_doc = create_problem_definition_document(
    business_context,
    ml_problem_type,
    success_metrics,
    constraints,
    stakeholders
)

print(problem_definition_doc)
```

## Group Exercise: Defining an ML Problem

### Instructions
In small groups, analyze the following business scenario and develop a comprehensive problem definition.

### Business Scenario: Retail Product Recommendation

A retail company wants to increase average order value by recommending relevant products to customers during the online checkout process. They have historical purchase data, customer browsing behavior, and product information. The company wants to implement a recommendation system that suggests products customers are likely to add to their cart.

### Questions to Address:
1. What is the specific business goal? How will success be measured?
2. What type of ML problem is this? What is the target variable?
3. What data will be needed? What features might be relevant?
4. What are the key constraints and requirements?
5. Who are the key stakeholders and what are their needs?

## Key Takeaways for Business Applications

1. **Start with the Business Goal**: A well-defined ML problem always begins with a clear understanding of the business objective it aims to address.

2. **Be Specific and Measurable**: Define concrete success metrics that directly connect model performance to business impact.

3. **Understand Constraints Early**: Identify technical, data, business, and ethical constraints at the beginning to avoid wasted effort.

4. **Involve All Stakeholders**: Ensure alignment between business stakeholders, end users, and the technical team throughout the problem definition process.

5. **Document Thoroughly**: Create a comprehensive problem definition document that serves as a reference throughout the project lifecycle.

A well-defined problem is half-solved. By investing time in proper problem definition, you set the foundation for a successful machine learning project that delivers real business value.
