# 💳 Lending Club Risk Analytics & Pricing Optimization Dashboard

An interactive web application for credit risk assessment and risk-based loan pricing. This project applies advanced Data Science workflows to predict the optimal interest rate for borrowers based on their financial and credit risk profiles.

🚀 **Live Demo:** [View Streamlit Web App](https://lendingclub-analytics.streamlit.app/)
📊 **Full Dataset:** [Lending Club Dataset on Kaggle](https://www.kaggle.com/datasets/fuong123/data-bai1/)
---

## 📌 Project Overview

When a financial institution lends money, they must balance risk and return. This project uses historical peer-to-peer lending data from Lending Club to build an automated risk assessment system. 

The core business logic is **Risk-Based Pricing**:
* **Low-Risk Borrowers:** Get competitive, low interest rates to encourage borrowing and ensure customer retention.
* **High-Risk Borrowers:** Charged a higher **Risk Premium** (higher interest rates) to cover expected defaults and protect the bank's capital.

---

## 🔬 Deep Dive: Notebook Analytics Workflow

The core of this project lies in the rigorous data processing and modeling conducted within the research notebook. Below is the detailed breakdown of the technical pipeline:

### 1. Advanced Feature Engineering & Data Preprocessing
* **FICO Score Consolidation:** Combined `fico_range_low` and `fico_range_high` into a single, robust `fico_score` feature representing the borrower's overall credit worthiness.
* **Mathematical Log-Transformation:** Financial variables like `annual_inc` (Annual Income) usually have severe right-skewness due to extreme outliers (high-income individuals). The notebook applies a $Log_{10}(X + 1)$ transformation to pull these outliers closer, stabilizing variable variance and improving linear model coefficients.
* **Categorical Encoding:** Converted string features like loan `term`, `purpose`, and `home_ownership` into structured formats suitable for model ingestion.

### 2. Statistical Testing & Multicollinearity Control
* **Pearson Correlation Analysis:** Generated a comprehensive correlation heatmap to map the linear relationships between risk metrics.
* **Multicollinearity Discovery:** The analysis revealed a near-perfect linear relationship ($r \approx 0.95$) between `loan_amnt` (Loan Amount) and `installment` (Monthly Payment). To prevent high Variance Inflation Factor (VIF) values from distorting model coefficients, feature selection strategies were implemented.
* **Dimension Reduction via PCA:** Applied **Principal Component Analysis (PCA)** on core financial drivers (`loan_amnt`, `annual_inc`, `dti`) to project the data into a 2D space (PC1 & PC2). This proved that clear clusters of risk levels exist even when dimensionality is reduced.

### 3. Machine Learning Architecture & Benchmarking
The notebook tests a diverse range of algorithms to evaluate performance trade-offs between simple baselines and complex ensembles:
* **Ridge Regression (Linear Baseline):** Used L2 regularization to minimize overfitting and evaluate baseline linear performance.
* **Random Forest (Bagging Ensemble):** Deployed on a sampled subset (100k rows) due to local RAM boundaries, offering strong non-linear boundary separation.
* **XGBoost (Gradient Boosting Engine):** The champion model accelerated via GPU training. It handles non-linear structures and missing values natively, achieving peak accuracy.

---

## 📊 Model Performance Evaluation

All models were evaluated on an independent test split using standard quantitative metrics:

| Evaluation Metric | XGBoost Engine | Ridge Baseline | Random Forest |
| :--- | :---: | :---: | :---: |
| **R-squared ($R^2$) (↑)** | **0.9771** | 0.4621 | 0.6142 |
| **Mean Absolute Error (MAE) (↓)** | **0.4706%** | 0.6859% | 2.1734% |
| **Root Mean Squared Error (RMSE) (↓)** | **0.7301%** | 0.9385% | 2.9992% |

### Key Technical Insights from Results:
1. **XGBoost Superiority:** XGBoost dominates the leaderboard with an $R^2$ of **97.71%** and an error rate ($\text{MAE}$) below **0.5%**. This means the model's recommended interest rate matches real-world pricing with extreme precision.
2. **The Sampling Limitation:** Random Forest shows underfitting behavior because local hardware limits forced a downsampling of the data. This highlights the value of XGBoost's efficient memory management.

---

## 🖥️ Streamlit Dashboard Application

To make the notebook's findings usable for business operations, the pipeline was deployed as an production-ready application featuring:
* **Dynamic Bi-Theme Visualizations:** Built using **Plotly Express** to ensure charts instantly adjust and remain 100% readable with sharp black text in Light Mode and bright white text in Dark Mode.
* **Production Memory Management:** Uses `@st.cache_data` to load a representational slice (`data_sample.csv`) for fast rendering, avoiding server memory leaks.
* **Real-time Automated Underwriting Form:** Allows risk officers to input customer metrics manually and receive an instant interest rate decision (Auto-Approved, Manual Review, or Rejected).

---

## 🛠️ Tech Stack & Key Libraries

* **Dashboard Frontend:** `Streamlit`, `Plotly Express` (Interactive charts).
* **Core Analytics & Math:** `Pandas`, `NumPy`, `Scikit-Learn`, `Statsmodels`.
* **Model Inference Engine:** `XGBoost`, `Joblib` (Model serialization).
* **Static Visuals:** `Matplotlib`, `Seaborn`.

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone [https://github.com/Nhaphuong05/Lending_club.git](https://github.com/Nhaphuong05/Lending_club.git)
cd Lending_club
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Launch the App
```bash
streamlit run app.py
```
---
## 📁 Repository Structure
```
├── notebook.ipynb         # Jupyter Notebook containing full EDA, feature engineering, and model training
├── app.py                 # Core Streamlit app handling dashboard UI and ML prediction
├── requirements.txt       # Dependencies for cloud environment setup
├── .gitignore             # Shields large 1.6 GB raw data files from repository limits
├── data_sample.csv        # Scaled-down dataset sample for fast dashboard chart plotting
├── xgboost_model.pkl      # Serialized tree boosting weights ready for deployment
└── scaler.pkl             # Pre-trained Scikit-Learn pipeline for real-time inference normalization
```
