
# ML Assignment 2 – Classification Models Deployment

## a. Problem Statement
The objective of this assignment is to implement multiple machine learning classification models on a single dataset, evaluate their performance using standard metrics, and deploy the models through an interactive Streamlit web application.

## b. Dataset Description
The dataset used is the Heart Disease UCI dataset obtained from a public repository.
- Type: Binary Classification
- Number of instances: 920
- Number of features: 15
- Target variable: num (converted to binary: 0 = No disease, 1 = Disease)

## c. Models Used and Evaluation Metrics

The following six classification models were implemented and evaluated on the same dataset:

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | (from output) | (from output) | (from output) | (from output) | (from output) | (from output) |
| Decision Tree | (from output) | (from output) | (from output) | (from output) | (from output) | (from output) |
| KNN | (from output) | (from output) | (from output) | (from output) | (from output) | (from output) |
| Naive Bayes | (from output) | (from output) | (from output) | (from output) | (from output) | (from output) |
| Random Forest | (from output) | (from output) | (from output) | (from output) | (from output) | (from output) |
| XGBoost | (from output) | (from output) | (from output) | (from output) | (from output) | (from output) |

## d. Observations on Model Performance

| ML Model | Observation |
|--------|-------------|
| Logistic Regression | Performed well on linearly separable data but struggled with complex patterns |
| Decision Tree | Showed tendency to overfit the training data |
| KNN | Performance was sensitive to feature scaling |
| Naive Bayes | Fast and efficient but limited by independence assumption |
| Random Forest | Delivered strong and stable performance due to ensemble learning |
| XGBoost | Achieved the best overall performance with high predictive accuracy |
