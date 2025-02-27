<h1>
  <span class="headline">ML Workflow and Best Practices</span>
  <span class="subhead">Data Preprocessing</span>
</h1>

**Learning objective:** By the end of this lesson, students will be able to describe the data preprocessing process in the machine learning workflow.

## An Introduction to Data Preprocessing

Data preprocessing involves preparing raw data for analysis and modeling. This step ensures that the data is clean, consistent, and structured in a way that enhances the performance of machine learning models. Since raw data often contains noise, inconsistencies, or irrelevant information, preprocessing is a critical step in the ML workflow.

## Key Steps in Data Preprocessing

| **Preprocessing Step**          | **Description**                                                                                              | **Techniques**                                                                                                 | **Example**                                                |
|----------------------------------|--------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|
| **Handling Missing Data**       | Missing values can distort model performance if not addressed properly.                                       | - **Removal**: Drop rows or columns (use sparingly). <br> - **Imputation**: Mean/Median for numerical data, Mode for categorical data, advanced methods like KNN or regression. | Replace missing age values in a dataset with the median age.|
| **Removing Duplicates**          | Duplicate entries can bias the model and inflate certain patterns.                                            | - Remove duplicate rows to ensure data integrity.                                                             | Eliminate repeated entries in a customer transaction dataset. |
| **Handling Outliers**            | Outliers can skew model predictions and reduce accuracy.                                                     | - **Z-score Method**: Identify values outside ±3 standard deviations. <br> - **IQR Method**: Remove values outside 1.5×IQR. <br> - **Capping/Clipping**: Replace extreme values with limits. | Cap unusually high salary values to avoid skewed regression models. |
| **Scaling and Normalization**    | Ensures features are on the same scale to prevent models from being biased by larger-scale variables.         | - **Standardization**: Mean = 0, SD = 1. <br> - **Min-Max Scaling**: Scale to range [0, 1]. <br> - **Log Transformation**: Reduces skewness. | Normalize customer income and age for better comparability. |
| **Encoding Categorical Variables** | Converts categorical data into numerical format for machine learning models.                                 | - **Label Encoding**: Converts categories to integers. <br> - **One-Hot Encoding**: Creates binary columns for each category. <br> - **Ordinal Encoding**: Assigns ordered numerical values. | Convert "Low," "Medium," and "High" risk levels into 0, 1, and 2. |

## Challenges in Data Preprocessing:
1. **Data Volume**: Processing large datasets efficiently can be computationally intensive.
2. **Noise and Irrelevance**: Identifying and removing irrelevant data can be subjective.
3. **Domain Knowledge**: Requires an understanding of the domain to make informed preprocessing decisions.
4. **Overprocessing**: Excessive data manipulation can result in loss of valuable information.

## Best Practices for Data Preprocessing:

- 🕵️ **Understand the Data**: Perform initial exploration to identify potential issues.
- 📝 **Document the Process**: Keep track of all transformations for reproducibility.
- 🔄 **Iterative Refinement**: Revisit preprocessing steps as new insights emerge during modeling.