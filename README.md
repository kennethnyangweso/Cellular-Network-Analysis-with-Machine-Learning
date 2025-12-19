# 📡 Cellular Network Performance Analysis

This project uses real-world cellular network data to **analyze, model, and predict key network performance metrics** using machine learning. It focuses on regression tasks for network performance (**signal strength and throughput**) and classification for **network type** (3G, 4G, 5G, etc.) based on environmental and signal parameters.  

---

## 🎯 Project Overview

The main objectives of this project are:  

1. Predict **signal strength and throughput** using regression models.  
2. **Classify network type** (3G, 4G, 5G, etc.) using machine learning classification models.  
3. Analyze the impact of measurement tools, environmental factors, and signal parameters on network performance.  

---

## 💡 Problem Statement

Mobile network operators often experience **inconsistent service quality** across different environments. Challenges include:  

- Predicting network performance based on environmental and signal factors.  
- Identifying which factors most strongly influence network type classification.  
- Supporting decision-making for network optimization and deployment planning.  

This project uses machine learning to address these challenges.  

---

## 📊 Dataset Overview

**Dataset Source:** Real cellular network measurement data.  

**Key Features include:**  
- Signal strength (dBm)  
- Throughput (Mbps)  
- Latency (ms)  
- Measurement tool indicators  
- Environmental parameters (temperature, interference, etc.)  
- Time components (hour, day of week, month)  
- Network type (3G, 4G, 5G) — target for classification  



---

## 📈 Exploratory Data Analysis (EDA)

### Key Visualizations
 
 **Distribution of signal strength**  

<img width="784" height="384" alt="image" src="https://github.com/user-attachments/assets/81b2a828-d376-4391-93f6-136d493b90b2" />

 
 **Distribution of Throughput**

<img width="784" height="384" alt="image" src="https://github.com/user-attachments/assets/b9143f32-10c9-43a2-ba2b-cedb32899d46" />

 
 **Correlation heatmap** to find relationships between features  

 <img width="866" height="783" alt="image" src="https://github.com/user-attachments/assets/6d7b6380-f512-4d2d-b05d-1df6bc1832ec" />

 **Network type distribution**  

<img width="482" height="502" alt="image" src="https://github.com/user-attachments/assets/9319aa7e-271b-4b90-87fb-59668b56ff0b" />
  
#### **Key Insights**

- **Signal Strength (dBm)**: The distribution is approximately normal, centered around -90 dBm, which represents typical signal strengths within the dataset.
- **Data Throughput (Mbps)**: This feature is highly skewed towards lower values, indicating that most measurements show lower data speeds, with fewer instances of high throughput.
- **Near-Even Distribution for network types**: A key observation is that all four network types have a very similar, almost equal, distribution. Each category accounts for approximately 25% of the total, suggesting a nearly uniform distribution among these network types.

---

## 🧠 Modeling Approach

### 🔹 Regression Models
Predict continuous performance metrics:  
- Signal strength  
- Throughput  
 

**Algorithms used:**  
- Linear Regression  
- Random Forest Regressor  
- XGBoost Regressor  

**Evaluation Metrics:**  
- RMSE (Root Mean Squared Error)  
- MAE (Mean Absolute Error)  
- R² Score  

### 🔹 Classification Models
Predict **network type** (3G, 4G, 5G) from environmental and signal features.  

**Algorithms used:**  
- Random Forest  
- XGBoost  
 

**Evaluation Metrics:**  
- Accuracy  
- Precision  
- Recall  
- F1-Score  

---

## 🧪 Performance Metrics

### Regression

### Signal Strength best Model ( XGBoost Regressor)

### Throughput best Model ( XGBoost Regressor )

| Metric | Signal Strength | Throughput |
|--------|----------------|------------|
| RMSE   |  1.74           |  0.08        | 
| MAE    |  1.21           |  0.03        | 
| R²     |  0.89           |  0.99        |  

### Classification ( Best Model Random Forest Classifier )

| Metric  | Network Type |
|---------|--------------|
| Accuracy| 0.99          |
| Precision| 0.99          |
| Recall | 0.99           |
| F1-Score| 0.99          |

### **Key Insights**

- The regression models perform very well, with low errors and high R² values
- The Random Forest Classifier for network type achieved excellent performance across all metrics (Accuracy, Precision, Recall, F1-Score = 0.99), demonstrating that environmental and signal parameters are highly predictive of the network type.
- Overall, the models are robust and reliable for both regression and classification tasks, supporting effective network performance analysis and prediction.

## 📈 Visuals and Confusion Matrix

<img width="784" height="584" alt="image" src="https://github.com/user-attachments/assets/3c0e446b-392b-49d2-bdb4-b28950d9ad73" />

<img width="784" height="584" alt="image" src="https://github.com/user-attachments/assets/1dfbe8c3-a0d2-4b05-89fb-78f5b8210f18" />

<img width="531" height="437" alt="image" src="https://github.com/user-attachments/assets/b4642f62-b5b3-4f0c-900c-bc221da2dfe6" />

### **Key Observations**

1. Excellent Fit  points cluster tightly around the 45° line, confirming the high R² (0.8926).
2. Near-Perfect Alignment points closely follow the 45° diagonal, confirming the model’s exceptional accuracy (R² ≈ 0.9995).
3. The confusion matrix shows a model that achieved perfect classification across all four classes (LTE, 4G, 3G, 5G). 

---
## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter&logoColor=white)


### 📊 Data Visualization
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-4B8BBE?style=for-the-badge)

### 🚀 ML App Deployment
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
---



## 📦 Installation

Clone the repository and install dependencies:

bash

     git clone https://github.com/kennethnyangweso/Cellular-Network-Analysis-with-Machine-Learning.git
     cd Cellular-Network-Analysis-with-Machine-Learning
     python3 -m venv venv
     source venv/bin/activate  # On Windows use `venv\Scripts\activate`
     pip install -r requirements.txt


## 🚀 Usage / Deployment

🧠 Run Analysis Notebook

    jupyter notebook notebooks/Cellular_Network_Analysis.ipynb

🌐 Run Streamlit Deployment App

    streamlit run app.py

## **Expected Results**

### Signal Strength Prediction

![app - Google Chrome 6_16_2025 3_55_08 PM](https://github.com/user-attachments/assets/9a3e5f69-a35e-44f7-b539-bdedfa050ec9)



![app - Google Chrome 6_16_2025 3_55_27 PM](https://github.com/user-attachments/assets/face742f-72a7-4286-ae29-3f98b8190889)

### Throughout Prediction

![app - Google Chrome 6_16_2025 4_05_23 PM](https://github.com/user-attachments/assets/18fe8ff9-8288-4aa0-99bd-fa6897cefdfa)


![throughput - Google Chrome 6_16_2025 4_05_44 PM](https://github.com/user-attachments/assets/ef1717fa-60d4-475d-b638-b9dc7de18e9c)


### Network Type Prediction

![network - Google Chrome 6_16_2025 4_17_43 PM](https://github.com/user-attachments/assets/2d84d372-efc3-44e4-be40-69297197e481)


![network - Google Chrome 6_16_2025 4_18_00 PM](https://github.com/user-attachments/assets/efbbc45e-962a-4711-bb55-f4d295df2c7e)

---

## Licenses

![License](https://img.shields.io/github/license/kennethnyangweso/Cellular-Network-Analysis-with-Machine-Learning)
![Stars](https://img.shields.io/github/stars/kennethnyangweso/Cellular-Network-Analysis-with-Machine-Learning)





