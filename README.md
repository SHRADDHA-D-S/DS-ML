# DS-ML (Data Science & Machine Learning)

Welcome to the **DS-ML** repository. This project is a curated collection of Python-based data science and machine learning tasks, focusing on Exploratory Data Analysis (EDA), Feature Engineering, and predictive modeling.

---

## 📁 Repository Structure

Below is an overview of the key files in the repository:

*   **`EDA_project.py`**: A comprehensive Python script for performing Exploratory Data Analysis (EDA) on customer sales demographics.
*   **`Feature_Engineering.py`**: A Python script designed for implementing feature selection, scaling, and transformation workflows.
*   **`sales.csv` & `data.csv`**: Customer datasets containing key demographics and behaviors such as Customer ID, Age, City, Spending, and Monthly Visits. Used for testing the EDA and feature engineering pipelines.
*   **`README.md`**: Project documentation (this file).

---

## 🔍 Exploratory Data Analysis (EDA)

The `EDA_project.py` script performs the following steps:
1.  **Data Loading & Inspection**: Checks for dataset existence, loads the CSV using `pandas`, and inspects basic statistics (shape, head, tail, and summary statistics using `describe()`).
2.  **Missing Value Imputation**: 
    *   Fills missing values in the `Age` column with the **median age**.
    *   Fills missing values in the `Spending` column with the **mean spending**.
3.  **Data Visualization**:
    *   Plots a histogram to show the distribution of spending.
    *   Generates a correlation matrix and plots it as a heatmap using `seaborn` and `matplotlib`.
    *   Visualizes customer age distribution using a box plot to detect anomalies.
4.  **Outlier Detection**: Identifies and prints potential outliers (e.g., customers with an age greater than 100).

---

## 🛠️ Installation & Setup

To run the scripts locally, make sure you have Python installed, along with the required libraries.

### 1. Install Dependencies
You can install the necessary packages using `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

*(Optional)* If you plan to use target encoding for categorical features:
```bash
pip install category-encoders
```

### 2. Run the Scripts
Execute the scripts using the Python command line:

```bash
# To run the EDA analysis
python EDA_project.py

# To run the Feature Engineering workflow
python Feature_Engineering.py
```

---

## 🚀 Future Additions
We are currently expanding the project to include machine learning pipelines, starting with regression models on tabular data (e.g., baseball statistics), missing data imputation strategies, and advanced feature selection techniques (e.g., `SelectKBest` with mutual information regression).