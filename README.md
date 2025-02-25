# Combinational-complexity
AI algorithm to predict combinational logic depth in RTL designs. Reduces timing violations by predicting logic depth without synthesis. 
ory:

## Overview
This project predicts the **combinational logic depth** of signals in RTL designs using machine learning. It helps identify **timing violations** early, reducing the need for time-consuming synthesis. The model is trained on synthetic RTL data with features like **Fan-In**, **Fan-Out**, **Gate Count**, and **Gate Types**.

## Key Features
- **Dataset Creation**: Synthetic RTL datasets simulate real-world timing reports.
- **Feature Engineering**: Categorical features are encoded, and numerical features are normalized.
- **Model Training**: Multiple ML models (Random Forest, Gradient Boosting, XGBoost, SVR, Neural Network) are trained and evaluated.
- **Best Model**: Neural Network (MLPRegressor) with **R²: 0.72** and **MSE: 0.75**.
- **Fast Prediction**: Predicts logic depth in under 1 second.

## Repository Contents
- **Jupyter Notebook**: Code for dataset generation, feature engineering, model training, and evaluation.
- **Dataset**: Synthetic RTL dataset (`final_synthetic_rtl_dataset.csv`).
- **Preprocessing Pipeline**: Saved pipeline (`rtl_preprocessor.pkl`) for feature encoding and scaling.
- **PDF Document**: Summary of approach, proof of correctness, and complexity analysis.

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook:
   ```bash
   jupyter notebook googlefinal_final.ipynb
   ```

## Results
- **Best Model**: Neural Network (MLPRegressor).
- **Performance**:
  - **MSE**: 0.75
  - **R² Score**: 0.72
- **Feature Importance**: Gate Count and Fan-In are the most influential features.

## Applications
- **Early Timing Analysis**: Predict timing violations during RTL design.
- **Design Optimization**: Optimize signals with high logic depth early.
- **EDA Integration**: Can be integrated into EDA tools for real-time feedback.
