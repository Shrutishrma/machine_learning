# Machine Learning Classification Projects

This repository contains standalone machine learning implementations focusing on supervised classification tasks. The projects range from end-to-end data preprocessing pipelines to probabilistic modeling.

## 📂 Repository Contents

### 1\. 🏦 Loan Approval Prediction

**File:** `Loan_Approval_Prediction/loan_approval_prediction.ipynb`

This project implements a supervised learning model to predict bank loan eligibility. It features a robust data engineering pipeline to handle real-world data inconsistencies.

  * **Model:** Logistic Regression.
  * **Preprocessing Pipeline:** \* Used `ColumnTransformer` to apply different strategies to specific feature sets.
      * **Imputation:** Handled missing values using `median` for numerical data and `most_frequent` for categorical data.
      * **Encoding:** Applied `OneHotEncoder` for nominal variables and `OrdinalEncoder` for ranked geographic data.
      * **Scaling:** Integrated `StandardScaler` and `MinMaxScaler` within the transformation block.
  * **Optimization:** Utilized `GridSearchCV` for hyperparameter tuning, identifying the optimal configuration ($C=0.01, penalty='l2'$).
  * **Deployment Readiness:** The final model is serialized using `pickle` for integration into production environments.

### 2\. 🧠 Naive Bayes Classification

**File:** `Naive_Bayes_Classification/naive_bayes_classification.ipynb`

An implementation of a probabilistic classifier using the Gaussian Naive Bayes algorithm. This project demonstrates the efficiency of Bayesian logic for multi-class classification problems.

  * **Model:** `GaussianNB`.
  * **Dataset:** Iris Flower Dataset.
  * **Performance:** Achieved an accuracy of **\~97.7%**, demonstrating high predictive power with minimal computational overhead.

-----

## 🛠️ Technical Stack

  * **Language:** Python
  * **Data Manipulation:** Pandas, NumPy
  * **Machine Learning:** Scikit-Learn
  * **Feature Engineering:** `SimpleImputer`, `OneHotEncoder`, `StandardScaler`, `Pipeline`
  * **Model Selection:** `train_test_split`, `GridSearchCV`, `cross_val_score`

-----

## 🌍 Related Projects

For a comprehensive example of an end-to-end web-deployed machine learning application (including Web Scraping, Clustering, and a Django backend), please visit my dedicated repository:
**[Fertility Rate Prediction Hub](https://www.google.com/search?q=https://github.com/Shrutishrma/fertility-rate-prediction)**

-----
