# Bank Marketing Classification Project

## 📌 Project Overview
This project focuses on predicting whether a bank client will subscribe to a **term deposit** based on marketing campaign data.  
It applies **data preprocessing, exploratory data analysis (EDA), and machine learning classification models** to build a predictive pipeline.  

Dataset: [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)  

---

## 🎯 Objectives
- Understand client behavior through data analysis  
- Build machine learning models to predict term deposit subscription  
- Evaluate model performance and optimize for business applications  

---

## ⚙️ Tech Stack
- **Python**  
- **Pandas, NumPy** – Data manipulation  
- **Matplotlib, Seaborn** – Visualization  
- **Scikit-learn** – Machine learning, pipelines, evaluation  
- **Jupyter Notebook**  

---

## 📊 Workflow
1. **Data Preprocessing**
   - Handle missing values (`unknown`)  
   - Encode categorical features (OneHot, Ordinal)  
   - Scale numeric features  
   - Handle imbalanced data (stratified split)  

2. **Exploratory Data Analysis (EDA)**
   - Distribution of features  
   - Correlation analysis  
   - Insights into customer behavior  

3. **Modeling**
   - Logistic Regression  
   - Decision Tree  
   - K-Nearest Neighbors  

4. **Hyperparameter Tuning**
   - GridSearchCV  
   - Optimal threshold selection  

5. **Evaluation**
   - Accuracy, Precision, Recall, F1-score, ROC-AUC  
   - ROC Curves with optimal thresholds  

---

## 📈 Results
- Logistic Regression and Decision Tree performed well on imbalanced data  
- Optimal threshold tuning improved **recall**, which is crucial for identifying potential clients  
- Insights help banks **optimize marketing strategy** and **reduce costs**  

---

## 👤 My Contribution
- Built classification models (Logistic Regression, Decision Tree, KNN)  
- Implemented preprocessing pipelines with **ColumnTransformer**  
- Performed hyperparameter tuning (GridSearchCV)  
- Evaluated models and compared metrics  
- Visualized results (ROC curves, feature importance)  

---

## 🚀 How to Run
Clone the repo and open the Jupyter Notebook:

```bash
git clone https://github.com/<your-username>/bank-marketing-classification.git
cd bank-marketing-classification
jupyter notebook
