# E-Commerce-Customer-Churn-Prediction

Predicts which customers of an online retail company are likely to churn, so retention offers can be targeted at the right accounts instead of blasted to everyone.


## Data
5,630 customers, 20 raw features (tenure, order behavior, satisfaction score, complaints, payment mode, etc.)
~17% churn rate — moderate class imbalance, handled with class_weight='balanced' rather than SMOTE (simpler to explain, sufficient here)
Missingness in 7 numeric columns flagged as its own binary feature before median imputation, since "did this customer have missing data" was itself informative
Duplicate rows dropped, inconsistent category labels merged (e.g. "Phone" → "Mobile Phone")
Approach

Three models compared head-to-head on the same test set (80/20 stratified split):

Model	ROC-AUC	PR-AUC	Churn Precision	Churn Recall
Logistic Regression	0.879	0.684	0.447	0.804
Random Forest	0.941	0.791	0.602	0.842
Gradient Boosting	0.956	0.837	0.643	0.900


Gradient Boosting was selected for deployment. It catches 90% of actual churners while keeping the false-positive rate low enough (~10%) that a retention campaign built on its output wouldn't be mostly wasted on customers who were never leaving. Random Forest was explicitly rejected — it beat Logistic Regression but was dominated by Gradient Boosting on every single metric, so no further tuning time went into it.

## Key churn drivers (permutation importance)
Tenure
Complaint history
Number of addresses on file
Satisfaction score
Cashback amount
Stack

<img width="1006" height="600" alt="image" src="https://github.com/user-attachments/assets/f052a053-8b49-409a-8eea-7654d164e794" />


pandas · numpy · scikit-learn · matplotlib / seaborn / plotly


Structure
Customer_Churn_Prediction.ipynb   # full pipeline: EDA → preprocessing → modeling → evaluation
