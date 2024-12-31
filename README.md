# Bank Marketing Campaign Analysis - Python

## Overview
This project focuses on analyzing a dataset from a bank's marketing campaign to predict customer subscription to a term deposit. It employs machine learning techniques, including a Random Forest Classifier and a Neural Network, to develop robust predictive models.

## Objectives
- Analyze the factors influencing customer decisions.
- Build and evaluate machine learning models to predict term deposit subscriptions.
- Compare model performance to derive actionable insights.

## Dataset
- **Source:** UCI Machine Learning Repository - "Bank Marketing Dataset"
- **Dataset Details:**
  - File: `bank-additional-full.csv`
  - Rows: 41,188
  - Columns: 21
  - Target Variable: `y` (binary classification: "yes" or "no")

## Key Steps

### Data Preprocessing
1. **Handling Unknown Values:** Replace or retain "unknown" entries based on context.
2. **Imputation:** Fill missing values with appropriate measures (e.g., mode).
3. **Outlier Removal:** Use Z-scores to eliminate extreme values.
4. **Feature Scaling:** Standardize numerical features for consistency.
5. **One-Hot and Ordinal Encoding:** Convert categorical variables to numerical format.
6. **Balancing Classes:** Apply SMOTE to address target class imbalance.

### Exploratory Data Analysis (EDA)
- Feature distributions, correlations, and insights into customer behavior.
- Visualizations for categorical and numerical features.

### Model Development
- **Random Forest Classifier:**
  - Features: `max_depth=15`, `class_weight='balanced'`
  - Metrics: Precision, Recall, F1-Score, ROC-AUC
- **Neural Network Model:**
  - Architecture: Dense layers with dropout to prevent overfitting
  - Metrics: Accuracy, Loss, ROC-AUC

### Model Evaluation
- **Random Forest Results:**
  - Accuracy: 89.42%
  - ROC-AUC: 0.9582
  - Feature Importance: Economic indicators and customer demographics dominate.
- **Neural Network Results:**
  - Accuracy: 87.25%
  - ROC-AUC: 0.9484

## Results
- Both models demonstrate strong predictive performance.
- The Random Forest model outperforms the Neural Network in most metrics.

## Repository Structure
- `Bank_Marketing_CM2604.ipynb`: Complete analysis and model building.
- `README.md`: Project documentation.

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Bank_Marketing_MachineLearning_CW.git
   ```
   ```
3. Open the Google Colab:
   ```bash
   google colab Bank_Marketing_CM2604.ipynb
   ```

## Dependencies
- Python 3.x
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, tensorflow, imbalanced-learn

## Future Work
- Hyperparameter optimization.
- Explore advanced machine learning models (e.g., XGBoost).
- Develop an ensemble model combining Random Forest and Neural Network predictions.

## License
This project is licensed under the MIT License.

## Acknowledgments
- Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing)
- Inspiration: Coursework from CM2604 Machine Learning module.
- Special Gratitude to Module Team.
