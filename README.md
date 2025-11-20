# 📊 HR Employee Attrition Prediction


## 💡 Business Problem
Employee turnover (attrition) is a major cost for businesses. Replacing an employee can cost up to **2x their annual salary** due to recruiting, training, and lost productivity. 

**The Goal:** To build a predictive model that identifies employees at risk of leaving. By identifying these "at-risk" employees early, HR departments can intervene with retention strategies (e.g., salary adjustments, promotion discussions, or work-life balance initiatives).

---

## 📂 The Dataset
* **Source:** IBM HR Analytics Employee Attrition & Performance.
* **Size:** 1,470 Rows (Employees), 35 Columns (Features).
* **Target Variable:** `Attrition` (Yes/No).
* **Key Features:**
    * **Demographics:** Age, Gender, Marital Status.
    * **Job details:** Department, Job Role, Business Travel.
    * **Compensation:** Monthly Income, Percent Salary Hike, Stock Option Level.
    * **Satisfaction:** Environment, Job, and Relationship Satisfaction scores.

---

## 🛠️ Solution Methodology

This project follows a rigorous Data Science lifecycle:

### 1. Exploratory Data Analysis & Cleaning
* **Data Integrity:** Checked for null values (0 found) and duplicates (0 found).
* **Noise Removal:** Dropped columns with zero variance or no predictive power (`EmployeeCount`, `Over18`, `StandardHours`).
* **Normalization:** Applied **Log Transformation** (`np.log1p`) to `MonthlyIncome`. 
    * *Why?* Financial data is typically right-skewed. Normalizing it helps the Logistic Regression model converge faster and perform better.

### 2. Advanced Feature Selection (VIF)
Multicollinearity occurs when features are highly correlated (e.g., "Job Level" and "Monthly Income"). This confuses the model.
* **Technique:** Calculated the **Variance Inflation Factor (VIF)** for all features.
* **Action:** Dropped features with VIF > 10 to ensure model stability:
    * `PerformanceRating` (Highly correlated with Salary Hike)
    * `JobLevel` (Highly correlated with Income)
    * `TotalWorkingYears` (Highly correlated with Age)
    * `YearsInCurrentRole` & `YearsWithCurrManager` (Redundant time metrics)

### 3. Preprocessing Pipeline
We used a `ColumnTransformer` to automate preprocessing:
* **Categorical Data:** Transformed using **One-Hot Encoding** (e.g., `Department` → `Department_Sales`, `Department_R&D`).
* **Numerical Data:** Scaled using **StandardScaler** to ensure features like `Age` (range 18-60) don't overpower features like `JobSatisfaction` (range 1-4).

### 4. Modeling
* **Algorithm:** Logistic Regression.
* **Class Imbalance Handling:** Used `class_weight='balanced'`.
    * *Why?* In this dataset, only ~16% of employees leave. A standard model might just predict "Stay" for everyone and still get 84% accuracy. The `balanced` weight forces the model to pay more attention to the minority class (Attrition = Yes).

---

## 📈 Model Performance
The model was evaluated on a held-out test set (30% of data).

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Accuracy** | **~74.6%** | Overall percentage of correct predictions. |
| **ROC-AUC** | **~72.3%** | The model's ability to distinguish between "Leavers" and "Stayers". |
| **Recall (Class 1)** | **High** | The model successfully catches a majority of the employees who actually left. |
| **Precision (Class 1)** | **Moderate** | There are some false alarms (predicting someone will leave when they stay), which is an acceptable trade-off for retention efforts. |

---
