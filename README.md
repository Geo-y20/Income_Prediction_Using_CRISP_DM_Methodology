# Income Prediction Using CRISP-DM Methodology

## Overview
The goal of this project is to predict whether an individual's income exceeds 50K (>50K) or is less than or equal to 50K (<=50K). This binary classification problem involves using demographic and employment-related data, such as age, education, occupation, and hours worked, to estimate an individual’s income level. The outcome of this analysis can be used by companies, policymakers, and research institutions to better understand the socioeconomic factors affecting income distribution.

### Dataset: [UCI Adult (Census Income) Dataset](https://archive.ics.uci.edu/dataset/2/adult)

The dataset contains demographic and employment data for individuals and is used to predict income levels. Key features include age, education, marital status, occupation, and more.

## Methodology

This project follows the CRISP-DM methodology:

### 1. Business Understanding
- **Objective**: Predict whether an individual's income exceeds 50K based on demographic features.
- **Justification for Dataset**: The UCI Adult (Census Income) dataset captures various factors like age, education, work class, and occupation, making it suitable for income prediction.

### 2. Data Understanding
- **Dataset Features**: The dataset includes both numeric and categorical features such as `age`, `education`, `occupation`, `capital-gain`, etc.
- **Correlation Analysis**: Certain features like `education-num` and `capital-gain` show a correlation with income, indicating their predictive power.
- **Missing Values**: Columns like `workclass`, `occupation`, and `native-country` have missing values, which were handled by imputation.

### 3. Data Preparation
- **Handling Missing Data**: Imputation was done with "Unknown" for missing values in categorical features.
- **Skewness Treatment**: Log transformations were applied to features like `capital-gain` and `capital-loss` due to skewness.
- **Categorical Encoding**: Label Encoding was used for categorical features to prepare them for machine learning models.
- **Target Variable Encoding**: The target `income` was encoded into binary values (`>50K` = 1, `<=50K` = 0).

### 4. Modeling (with SMOTE)
- **Objective**: Train multiple machine learning models to classify income.
- **Models Trained**:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Gradient Boosting Classifier

- **Class Imbalance Handling**: SMOTE (Synthetic Minority Oversampling Technique) was applied to balance the dataset.

### 5. Evaluation
- **Evaluation Metrics**: Models were evaluated using accuracy, precision, recall, and F1-score.
- **Best Performing Model**:  
  Gradient Boosting achieved the highest F1-score (0.692) and strong recall (0.832), making it the final model selected for deployment.

### 6. Deployment
- **Model Application**: The Gradient Boosting model can be deployed as an API to predict income based on user inputs.
- **API Integration**: An API endpoint can be created to receive input from web applications and return a prediction.
- **Web Application**: The model can be integrated into a web-based dashboard for real-time predictions.

## Image: Model Comparison

![Model Comparison](https://github.com/Geo-y20/Income_Prediction_Using_CRISP_DM_Methodology/blob/main/Model%20Comparison.png)

## Full Documentation

For the detailed methodology and steps involved, refer to the [Income Prediction Using CRISP-DM Methodology PDF](assets/documents/Income_Prediction_Using_CRISP-DM_Methodology.pdf).

## Installation

To run this project, you need to have the following libraries installed:

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

## Usage

1. Clone the repository:

```bash
git clone <repository-url>
```

2. Run the model training script:

```bash
python income_prediction.py
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

This project successfully applies the CRISP-DM methodology to predict income levels using demographic data. The selected Gradient Boosting model shows excellent performance and can be deployed to help businesses, policymakers, and other stakeholders make informed decisions about income-based strategies.

## Future Work

- **Model Retraining**: Periodically retrain the model with updated data to adapt to shifts in income patterns.
- **Feature Expansion**: Experiment with additional features such as marital status or geographical data.
- **Real-time Integration**: Further work on integrating the model into a web-based dashboard for real-time predictions.
