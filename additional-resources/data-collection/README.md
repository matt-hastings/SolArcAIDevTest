<h1>
  <span class="headline">ML Workflow and Best Practices</span>
  <span class="subhead">Data Collection</span>
</h1>

**Learning objective:** By the end of this lesson, you will be able to describe the data collection process in the machine learning workflow.

## Steps in Data Collection

| **Step**| **Description**| **Examples/Details**|
|:--------|:---------------|:--------------------|
| **Identify Data Requirements** | Determine what data is needed to address the problem.| Consider features, granularity, and historical depth. Example: For a customer churn model, collect customer demographics, transaction history, and interaction logs. |
| **Data Gathering**            | Extract data from identified sources.| Sources could include databases, APIs, or third-party providers.|
| **Validate Data Quality**     | Ensure the collected data meets the necessary standards.| Check for missing values, inconsistent formats, outliers, or noise.|
| **Data Consolidation**        | Combine data from multiple sources if needed.| Example: Merging sales data with customer feedback data.|
| **Document the Data Collection Process** | Maintain records of data sources, extraction methods, and any preprocessing steps.| Ensures transparency and reproducibility of the data collection process.|

## Challenges in Data Collection

🔍 **Data Availability**: Limited access to required data or insufficient historical records.

⚠️ **Data Quality Issues**: Incomplete, inconsistent, or noisy data.

📊 **Volume and Complexity**: Handling large-scale data or diverse formats (e.g., text, images, videos).

💰 **Cost**: Licensing fees or expenses associated with data acquisition.


## Best Practices for Data Collection

🎯 **Prioritize Relevant Data**: Focus on data that directly impacts the problem's solution.

🌍 **Ensure Data Diversity**: Collect data that represents different scenarios to avoid bias.

🤖 **Automate Data Collection**: Use scripts, APIs, or tools to streamline repetitive data extraction tasks.

🔒 **Ensure Data Security**: Protect sensitive information using encryption and access controls.

⏳ **Monitor and Update**: Regularly collect updated data to maintain model performance over time.


## 🗣️ Discussion Activity: Evaluating and Improving Data Collection

### Objective
Test your understanding of the data collection process, challenges, and best practices by analyzing a hypothetical scenario and making recommendations.

### Scenario
You are tasked with building a machine learning model to predict the likelihood of patients developing a certain medical condition. The initial dataset contains:

1. Patient demographics (age, gender, etc.).  
2. Medical history and previous diagnoses.  
3. Lifestyle information (diet, exercise habits).  
4. Lab test results.

However, the data has several issues:
- Missing values in the lab test results.
- Inconsistent formats in the lifestyle information (e.g., "Exercise: Yes" vs. "Physical Activity: Regular").
- Data collected only from urban hospitals, leaving rural areas unrepresented.
- Limited historical records for patients under 18 years old.

At this stage, the developer identifies the requirement of Data Pre-Processing.

### Discussion Questions
#### Data Requirements:  
- What additional data would improve the model's performance? Why?
- How can you ensure the dataset is representative of the broader population?

#### Data Quality Issues:  
- What steps would you take to handle the missing lab test values and inconsistent formats?
- How would you address the bias introduced by the lack of data from rural areas?

#### Best Practices:  
- Suggest specific actions to automate the data collection process while ensuring data security and diversity.  
- What steps can be implemented to monitor and update the dataset regularly?

#### Reflection:  
- How does following best practices in data collection ensure the reliability and fairness of the final model?