
# Income Prediction Using CRISP-DM Methodology

## Overview

This project uses the CRISP-DM methodology to predict whether an individual's income exceeds 50K (greater than 50K) or is less than or equal to 50K. The classification is based on demographic and employment-related data, such as age, education, occupation, and hours worked. This model can help organizations, policymakers, and research institutions understand the socioeconomic factors affecting income distribution.

### Dataset: [UCI Adult (Census Income) Dataset](https://archive.ics.uci.edu/dataset/2/adult)

The dataset contains demographic and employment data for individuals and is used to predict income levels. Key features include age, education, marital status, occupation, and more.

## Methodology

This project follows the CRISP-DM methodology:

1. **Business Understanding**  
   The objective is to predict whether an individual’s income is above 50K based on demographic features. This information is essential for businesses and policymakers aiming to understand income disparities.

2. **Data Understanding**  
   - **Dataset Features**: The dataset includes both numeric and categorical features such as `age`, `education`, `occupation`, `capital-gain`, etc.
   - **Correlation Analysis**: Certain features like `education-num` and `capital-gain` show a correlation with income, indicating their predictive power.
   - **Missing Values**: Columns like `workclass`, `occupation`, and `native-country` have missing values, which were handled by imputation.

3. **Data Preparation**  
   - **Handling Missing Data**: Imputation was done with "Unknown" for missing values in categorical features.
   - **Skewness Treatment**: Log transformations were applied to features like `capital-gain` and `capital-loss` due to skewness.
   - **Categorical Encoding**: Label Encoding was used for categorical features to prepare them for machine learning models.
   - **Target Variable Encoding**: The target `income` was encoded into binary values (`>50K` = 1, `<=50K` = 0).

4. **Modeling (with SMOTE)**  
   Several models were trained to classify income, including:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM)
   - Gradient Boosting

   **Handling Class Imbalance**: SMOTE (Synthetic Minority Oversampling Technique) was applied to balance the dataset.

5. **Evaluation**  
   Models were evaluated using:
   - Accuracy
   - Precision
   - Recall
   - F1-Score

   **Best Performing Model**:  
   The Gradient Boosting Classifier outperformed other models in terms of F1-Score (0.692) and recall (0.832), making it the final model selected for deployment.

6. **Deployment**  
   The Gradient Boosting model can be deployed in a production environment:
   - **API Integration**: The model can be integrated into a web application or API for real-time income predictions.
   - **Monitoring**: Regular monitoring and retraining will be implemented to ensure model performance remains accurate over time.

## Installation

To run this project, you need to have the following libraries installed:

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

## Usage

1. Clone the repository:

```bash
git clone <https://github.com/Geo-y20/Income_Prediction_Using_CRISP_DM_Methodology.git>
```

3. For deployment, the model can be integrated into an API using Flask or FastAPI.

## Results

The final model selection based on SMOTE-enhanced training:

| Model             | Accuracy | F1-Score | Precision | Recall |
|-------------------|----------|----------|-----------|--------|
| Gradient Boosting | 0.823    | 0.692    | 0.593     | 0.832  |
| Random Forest     | 0.834    | 0.677    | 0.632     | 0.729  |
| SVM               | 0.790    | 0.659    | 0.539     | 0.846  |
| KNN               | 0.798    | 0.634    | 0.559     | 0.732  |
| Decision Tree     | 0.800    | 0.616    | 0.569     | 0.672  |
| Logistic Regression| 0.755   | 0.600    | 0.493     | 0.767  |

## Conclusion

The project successfully applies the CRISP-DM methodology to predict income levels using demographic data. The selected Gradient Boosting model shows excellent performance and can be deployed to help businesses, policymakers, and other stakeholders make informed decisions about income-based strategies.

## Future Work

- **Model Retraining**: Periodically retrain the model with updated data to adapt to shifts in income patterns.
- **Feature Expansion**: Experiment with additional features such as marital status or geographical data.
- **Real-time Integration**: Further work on integrating the model into a web-based dashboard for real-time predictions.
