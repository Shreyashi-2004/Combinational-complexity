# Combinational Complexity Prediction in RTL Designs

## Overview
This project predicts the **combinational logic depth** of signals in RTL (Register Transfer Level) designs using machine learning. The goal is to identify **timing violations** early in the design process, reducing the need for time-consuming synthesis. The model is trained on synthetic RTL data with features like **Fan-In**, **Fan-Out**, **Gate Count**, and **Gate Types**.

---

## Problem Statement
Timing analysis is a critical step in RTL design, but it is typically performed after synthesis, which is time-consuming. This delay can lead to project bottlenecks, especially when timing violations require architectural refactoring. This project aims to **predict combinational logic depth** during the RTL design phase, enabling designers to identify and address timing issues earlier in the design cycle.

---

## Solution

### 1. **Dataset Creation**
- A synthetic RTL dataset was created to simulate real-world timing reports.
- The dataset includes the following features:
  - **Fan-In**: Number of inputs to a signal.
  - **Fan-Out**: Number of outputs from a signal.
  - **Gate Count**: Number of logic gates in the combinational path.
  - **Gate Types**: Types of gates used (e.g., AND, OR, NAND, NOR, XOR).
  - **Logic Depth**: Target variable representing the number of logic levels in the combinational path.

### 2. **Feature Engineering**
- **Categorical Encoding**: Gate types were one-hot encoded to convert them into numerical features.
- **Normalization**: Numerical features (Fan-In, Fan-Out, Gate Count) were standardized using `StandardScaler` to ensure consistent scaling.
- **Feature Selection**: The most important features (e.g., Gate Count, Fan-In) were identified using permutation importance.

### 3. **Model Training**
- Multiple machine learning models were trained and evaluated:
  - **Random Forest**
  - **Gradient Boosting**
  - **XGBoost**
  - **Support Vector Regressor (SVR)**
  - **Neural Network (MLPRegressor)**
- The **Neural Network (MLPRegressor)** performed the best and was selected as the final model.

### 4. **Model Evaluation**
- The model was evaluated using **Mean Squared Error (MSE)** and **R² Score**.
- **Best Model Performance**:
  - **MSE**: 0.76
  - **R² Score**: 0.69
- **Cross-Validation**: The model was validated using 5-fold cross-validation, achieving a mean R² score of **0.72**.

### 5. **Hyperparameter Tuning**
- The Neural Network was fine-tuned using **GridSearchCV** to optimize hyperparameters such as:
  - **Hidden Layer Sizes**: (100,)
  - **Activation Function**: ReLU
  - **Solver**: SGD
  - **Alpha (Regularization)**: 0.0001
  - **Max Iterations**: 2000

### 6. **Feature Importance**
- The most influential features for predicting logic depth were:
  - **Gate Count**: 0.9486
  - **Fan-In**: 0.3357
  - **Gate Types**: AND, OR, NAND, NOR, XOR

---

## Repository Contents
- **Jupyter Notebook**: Code for dataset generation, feature engineering, model training, and evaluation.
- **Dataset**: Synthetic RTL dataset (`final_synthetic_rtl_dataset.csv`).
- **Preprocessing Pipeline**: Saved pipeline (`rtl_preprocessor.pkl`) for feature encoding and scaling.
- **PDF Document**: Summary of approach, proof of correctness, and complexity analysis.

---

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

---

## Results
- **Best Model**: Neural Network (MLPRegressor).
- **Performance**:
  - **MSE**: 0.76
  - **R² Score**: 0.69
- **Prediction Runtime**: Less than 1 second per signal.

---

## Applications
- **Early Timing Analysis**: Predict timing violations during RTL design.
- **Design Optimization**: Optimize signals with high logic depth early.
- **EDA Integration**: Can be integrated into EDA tools for real-time feedback.

---

## Proof of Correctness
- The model was validated against a test dataset, and the predicted logic depths were compared to the actual depths from synthesis reports.
- The results show a strong correlation between predicted and actual logic depths, with an R² score of **0.69**.


---

