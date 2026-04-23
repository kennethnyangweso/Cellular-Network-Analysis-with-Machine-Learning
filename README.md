<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/d416e02d-7b92-44a5-80c5-ad7a7488ef89" />

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-006699?logo=plotly&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-2E8B57)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)


This project uses real-world cellular network data to **analyze, model, and predict key network performance metrics** using machine learning. It focuses on regression tasks for network performance (**signal strength and throughput**) and classification for **network type** (3G, 4G, 5G, etc.) based on environmental and signal parameters.  

---

## 🎯 Project Overview

The main objectives of this project are:  

1. Predict **signal strength** using regression models.  
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

## 🧠 Machine Learning Workflow
- Data cleaning & preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature engineering  
- Model training with multiple algorithms  
- Evaluation and comparison  
- Selection of the best performing models  

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

 

**Algorithms used:**  
- Decision Tree Regressor 
- Random Forest Regressor  
- XGBoost Regressor
- Hyperparameter Tuning
- GridSearch CV



**Evaluation Metrics:**  
- MSE (Root Mean Squared Error)  
- MAE (Mean Absolute Error)  
- R² Score  

### 🔹 Classification Models
Predict **network type** (3G, 4G, 5G) from environmental and signal features.  

**Algorithms used:** 
- Decision Tree Classifier
- Random Forest  Classifier
- XGBoost  Classifier 
- Hyperparameter Tuning
- GridSearch CV
 

**Evaluation Metrics:**  
- Accuracy
- F1-Score
- Classification Report

---

## 🧪 Performance Metrics

### Regression

### Signal Strength 



| Metric | Decision Tree |  **Random Forest** | XGBoost |
|--------|----------------|-------------------|---------|
| MAE (dBm)  |  1.34      |  **1.25**       | 1.27      |
| MSE (dBm)   |  3.40     |  **3.11**        | 3.17     |
| R² (%)    |  87.96      |  **88.98**      |  88.78    |

### Classification 

### Network Type 

| Metric  | Decision Tree | **Random Forest** | XGBoost       |
|---------|--------------|--------------------|---------------|
| Accuracy(%)| 87.73      |        **88.24**     |   88.11    |
| F1-Score (%)| 88.24     |     **88.13**       |     88.10   |


### 📊 Regression Insights – Signal Strength
- The Random Forest Regressor achieved the best overall performance, with the lowest MAE (1.25 dBm) and MSE (3.11), indicating more accurate and consistent predictions.
- It also recorded the highest R² score (88.98%), meaning it explains slightly more variance in signal strength compared to the other models.
- XGBoost performed very closely to Random Forest, making it a strong alternative, especially in scenarios where boosting may generalize better.
- The Decision Tree model showed slightly lower performance, likely due to overfitting and limited generalization capability.
👉 Conclusion: Random Forest is the most reliable model for signal strength prediction in this project.

### 📡 Classification Insights – Network Type
- The Random Forest Classifier achieved the highest accuracy (88.24%), indicating the best overall classification performance.
- Interestingly, the Decision Tree recorded the highest F1-score (88.24%), suggesting a slightly better balance between precision and recall for certain classes.
- XGBoost performed almost identically to Random Forest, confirming strong and stable classification capability across ensemble methods.
- The performance gap between all models is minimal, indicating that the dataset is well-structured and separable.
👉 Conclusion: While all models perform well, Random Forest offers the best balance between accuracy and consistency for network type classification.

## 📈 Visuals and Confusion Matrix For Best 

<img width="984" height="584" alt="image" src="https://github.com/user-attachments/assets/7de856c5-93df-403d-a337-73af643d75eb" />


<img width="656" height="550" alt="image" src="https://github.com/user-attachments/assets/cb627c9d-7b59-4fa5-8e70-b5e3745e8e0e" />


<img width="984" height="584" alt="image" src="https://github.com/user-attachments/assets/480aec70-dec5-4f9e-b179-4710cd339afe" />

### **Key Observations**

1.  Overall Good Fit: Most points are clustered tightly around the horizontal zero line, which confirms why your r-squared is so high (~0.89). For most of the signal range, the model is unbiased.
2.  Dominant Performance on Legacy Networks: The model remains perfect (100%) for 3G and 4G, showing zero confusion between older tech and modern LTE/5G signals.
3. Weighted Accuracy for LTE: The model is much more successful at identifying LTE (726 correct) than 5G (568 correct). It seems the tuning pushed the model to be more confident in its LTE predictions.
4. Shift to Throughput: Unlike the single Decision Tree (which ignored it), the Random Forest identified data_throughput_(mbps) as the #1 most important feature (~45%). This is likely what helped it achieve that record 88.2% accuracy, as 5G and LTE throughput differ significantly.
5.  A More Balanced Trio: The model relies on a powerful combination of Throughput, Latency, and Signal Strength. By using all three, it’s much harder for a signal to "hide" its true identity than if the model just looked at latency alone.
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

## **👤 Author**

**Kenneth Nyangweso**

**Data Scientist | Electrical & Telecommunications Engineer**




