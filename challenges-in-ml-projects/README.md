<h1>
  <span class="headline">ML Workflow and Best Practices</span>
  <span class="subhead">Challenges in ML Projects</span>
</h1>

**Learning objective:** By the end of this lesson, you'll be able to describe common challenges in machine learning projects and how to address them. 

## Examples of Challanges in ML Projects

| **Example**                          | **Challenges**                                                                                                   | **Proposed Solutions**                                                                                                                                           |
|--------------------------------------|-----------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Predictive Maintenance**           | - **Accuracy**: Imbalanced dataset as failures are rare, leading to a high risk of false negatives.             | - Use simple undersampling of the majority class to balance the dataset and improve failure detection accuracy.                                                  |
|                                      | - **Latency**: Predictions must be made within seconds to trigger real-time maintenance alerts.                 | - Deploy lightweight models like Logistic Regression or Decision Trees for faster predictions.                                                                  |
|                                      | - **Cost**: Sensor data collection and processing for thousands of machines can be computationally expensive.   | - Downsample data by aggregating sensor readings to reduce storage and compute costs.                                                                           |
| **Fraud Detection**                  | - **Accuracy**: Fraudulent transactions are rare compared to legitimate ones, causing class imbalance.          | - Apply a weighted classification approach (e.g., increasing the penalty for misclassifying the minority class) to address imbalance.                           |
|                                      | - **Latency**: Predictions must be generated within 100 milliseconds to avoid delays in transaction processing. | - Use a Random Forest model with fewer estimators to balance speed and accuracy.                                                                                |
|                                      | - **Cost**: Managing and storing large amounts of transaction data increases costs.                             | - Pre-filter transactions with simple rule-based heuristics before applying the ML model.                                                                       |
| **Customer Churn Prediction System** | - **Accuracy**: Imbalanced data as most customers are retained, while only a small percentage churn.            | - Use oversampling of the minority class (churned customers) by duplicating examples within the training set.                                                   |
|                                      | - **Latency**: Predictions must fit within a weekly ETL processing window.                                      | - Train a Decision Tree or Random Forest for interpretable and moderately accurate results.                                                                      |
|                                      | - **Cost**: Scaling storage and compute as the customer base grows.                                             | - Aggregate customer behavior data into summary statistics (e.g., average transactions) to reduce storage requirements.                                         |

---

## Addressing Key Machine Learning Challenges

#### 1.🧠 Understanding the Problem:
- **Define the Objective**:  
  Clearly specify the ML model's goal, such as classification, regression, or recommendation.  
- **Determine Success Metrics**:  
  - 🎯 Accuracy: Metrics like precision, recall, and F1 score.  
  - ⏱️ Latency: Time required to produce predictions.  
  - 💰 Cost: Resources for computation, data acquisition, and maintenance.

#### 2.🔍 Analyzing Accuracy Challenges:
- **Data Quality**:  
  - 📉 Are there missing values, outliers, or biases in the dataset?  
  - 🌍 Is the dataset representative of real-world scenarios?  
- **Model Limitations**:  
  - 🤔 Can the algorithm handle the complexity of the data?  
  - ⚠️ Are there risks of overfitting or underfitting?  
- **Evaluation Metrics**:  
  - ✅ Do chosen metrics align with business objectives?

#### 3.⚡ Evaluating Latency Challenges:
- **Prediction Speed**:  
  - ⏩ Is real-time or batch processing required?  
- **Model Complexity**:  
  - 🖥️ Is the model too resource-heavy for the deployment environment?  
  - 🔧 Can optimizations like quantization or pruning improve speed?  
- **System Integration**:  
  - 📡 Are delays introduced by API latency or network bottlenecks?

#### 4.💸 Assessing Cost Challenges:
- **Infrastructure Requirements**:  
  - 🔋 What hardware or cloud resources are necessary for training and inference?  
- **Model Development**:  
  - 🔄 How many iterations are needed to achieve acceptable accuracy?  
- **Scalability**:  
  - 📈 Can the solution scale with increasing data or traffic?  
- **Maintenance**:  
  - 🔧 What are the costs of retraining and updating the model?
  
#### 5.🔁 Iterative Problem Refinement:
- **Trade-Off Analysis**:  
  - ⚖️ Balance between accuracy, latency, and cost.  
- **Experimentation**:  
  - 🔬 Test different architectures, hyperparameters, and deployment strategies.  
- **Feedback Loop**:  
  - 🗣️ Use user and stakeholder feedback to refine priorities and solutions.



## 🗣️ **Discussion Activity 1**: Ways to Address Challenges in Accuracy, Latency, and Cost at ShopSmart

1. **Accuracy Challenges**:
   - **Problem**: ShopSmart's product recommendation system often suggests irrelevant products to customers, leading to low engagement and poor customer retention.
   - **Solutions**:
     - Enhance data quality by cleaning customer transaction records and removing outdated or incorrect data.
     - Use hybrid recommendation systems combining collaborative filtering and content-based filtering to improve relevance.
     - Perform feature engineering to include behavioral data, such as purchase frequency and average spend per visit, to make personalized recommendations.
     - Regularly validate and update models using real-world feedback from customer interactions.

2. **Latency Challenges**:
   - **Problem**: Real-time recommendations on ShopSmart's website take too long to load, causing user frustration and abandoned sessions.
   - **Solutions**:
     - Precompute frequently accessed recommendations and cache them to reduce response times.
     - Optimize model size using techniques like pruning or quantization for faster inference.
     - Deploy edge computing to bring computation closer to users, reducing latency.
     - Use lightweight models such as approximate nearest neighbor (ANN) algorithms for similarity searches.

3. **Cost Challenges**:
   - **Problem**: High computational costs for training and running ShopSmart's recommendation engine due to large-scale customer data.
   - **Solutions**:
     - Use pre-trained models or transfer learning to reduce the need for resource-intensive training.
     - Implement batch processing for non-real-time tasks, such as updating customer segmentation models.
     - Leverage cloud computing platforms with auto-scaling to optimize costs during peak shopping seasons.
     - Store data in compressed formats and use cost-effective storage solutions for historical customer data.



## 🗣️ **Discussion Activity 2**: Proposing Solutions for Real-World Challenges at ShopSmart

**Activity Description**:
Participants will analyze ShopSmart's challenges and propose solutions to improve its recommendation system.

**Scenario**:
ShopSmart faces the following challenges:
- Recommendations often lack accuracy, leading to irrelevant product suggestions.
- Real-time recommendations on their website are slow, causing poor user experience.
- Maintaining the recommendation system incurs high infrastructure and computational costs.

**Steps**:
1. **Identify the Challenges**:
   - Accuracy: Recommendations fail to reflect individual customer preferences.
   - Latency: Recommendations take more than 5 seconds to load during peak hours.
   - Cost: Monthly computational costs exceed the allocated budget.

2. **Propose Solutions**:
   - **For Accuracy**:
     - Enhance recommendations by including features such as Recency, Frequency, and Monetary Value (RFM analysis).
     - Incorporate preferred product categories into the model through feature engineering.
   - **For Latency**:
     - Precompute and cache recommendations for popular products and customers.
     - Deploy serverless architectures for dynamic scaling during high-traffic periods.
   - **For Cost**:
     - Reduce training frequency by implementing incremental learning for model updates.
     - Optimize storage by archiving old customer data in cheaper storage solutions.
   
3. **Present Solutions**:
   - Groups present their proposed solutions, emphasizing their impact on customer experience, system performance, and cost savings.


### Conclusion for ShopSmart

- **Summary of Challenges**:
  - ShopSmart's recommendation system struggles with accuracy, latency, and cost, impacting customer experience and operational efficiency.
