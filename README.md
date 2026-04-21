# 🧠 Customer Segmentation using RFM & Behavioral Analysis

# 🚀 Customer Segmentation App
[![Streamlit App](https://customersegmentationml-jgqdhdynktg9efdxtuctcq.streamlit.app/)

## 📌 Project Overview

This project focuses on **customer segmentation** using **RFM (Recency, Frequency, Monetary)** and additional behavioral features. The goal is to group customers based on their purchasing behavior and generate actionable business insights for targeted marketing and retention strategies.

---

## 🎯 Objectives

* Analyze customer purchasing patterns
* Segment customers into meaningful groups
* Identify high-value and at-risk customers
* Enable data-driven business decisions

---

## 📊 Dataset Features

The dataset contains pre-aggregated customer-level data with the following features:

* CustomerID
* TotalTransactions (Frequency)
* TotalSpending (Monetary)
* FirstPurchase, LastPurchase
* CustomerLifetime
* AvgOrderValue
* PurchaseFrequency
* TotalQuantity

---

## 🔧 Feature Engineering

* **Recency** calculated from LastPurchase
* **Frequency** derived from TotalTransactions
* **Monetary** derived from TotalSpending
* Combined RFM with behavioral features for deeper insights

---

## 🤖 Machine Learning Approach

### 🔹 Algorithms Used

* K-Means Clustering (primary segmentation)
* DBSCAN (outlier detection)

### 🔹 Feature Scaling

* StandardScaler applied to normalize feature values

### 🔹 Model Evaluation

* Silhouette Score
* Davies-Bouldin Index
* Calinski-Harabasz Score

---

## 📈 Customer Segments Identified

| Segment                | Description                                    |
| ---------------------- | ---------------------------------------------- |
| ⭐ VIP Customers        | High frequency, high spending, recent activity |
| 🛒 Regular Customers   | Moderate activity and spending                 |
| ⚠️ At-Risk Customers   | Previously active but inactive recently        |
| 💰 Low-Value Customers | Low engagement and low spending                |

---

## 📊 Visualization

* PCA-based 2D cluster visualization
* Scatter plots for cluster interpretation
* Segment-wise distribution analysis

---

## 🚀 Deployment

### 🔹 Streamlit Web App

* Interactive UI for inputting customer data
* Real-time prediction of customer segment
* Visual representation of clusters

### 🔹 Features

* Input-based prediction
* Business insights for each segment
* Cluster visualization using PCA

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* Streamlit
* Joblib

---

## 📂 Project Structure

```
├── app.py
├── customer_segmentation_model.pkl
├── scaler.pkl
├── Customers.csv
├── README.md
```

---

## 🔥 Key Learnings

* Applied unsupervised learning for real-world business problems
* Understood trade-offs between model performance and business usability
* Built end-to-end ML pipeline from preprocessing to deployment
* Gained experience in model evaluation and visualization

---

## 💡 Business Impact

* Enables targeted marketing strategies
* Helps retain at-risk customers
* Improves customer lifetime value
* Supports personalized recommendations

---

## 📌 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧠 Conclusion

This project demonstrates how machine learning can be used to transform raw customer data into actionable business insights through effective segmentation and deployment.

---
