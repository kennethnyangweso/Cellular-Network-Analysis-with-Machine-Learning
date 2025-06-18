# Cellular-Network-Analysis-with-Machine-Learning

##  Overview

In the telecommunications industry, maintaining high-quality cellular network performance is critical for customer satisfaction, retention, and operational efficiency. This project uses a real-world dataset containing geolocated signal measurements and network performance metrics to analyze and model key indicators of network health. By combining regression and classification approaches, the aim is to evaluate and predict network performance and classify zones based on service quality.

---

## Problem Statement

Mobile network operators face challenges in ensuring consistent service quality across regions. Variability in signal strength, data throughput, and latency can result in user dissatisfaction, especially in high-traffic or underserved areas. There is a need to:

- Accurately predict network performance metrics at any given location and time.
- Identify and classify zones by quality of service.
- Predict the network type (3G, 4G, 5G, etc.) based on environmental and signal parameters.

---

##  Project Objectives

### Primary Objectives

1. **Regression**:
   - Predict continuous performance metrics:
     - Signal Strength (dBm)
     - Data Throughput (Mbps)
     

2. **Classification**:
   - Classify zones into service quality levels: **Good**, **Moderate**, or **Poor**.
   - Predict the **network type** (e.g., 3G, 4G, 5G) based on observed signal and performance data.



###  Secondary Objectives

- Compare the reliability of different signal measurement tools (BB60C, srsRAN, BladeRFxA9).
- Understand correlation between geographic, temporal, and signal-related factors.

---

##  Metrics of Success

###  Technical Success Metrics

| Task | Metrics |
|------|---------|
| Regression (Signal/Throughput/Latency) | RMSE, MAE, R² |
| Zone Classification | Accuracy, Precision, Recall, F1-Score |
| Network Type Classification | Accuracy, Confusion Matrix, ROC-AUC |
| Feature Importance | Gain-based or SHAP value insights |

###  Business Success Metrics

| Goal | Indicator |
|------|-----------|
| Improved coverage planning | Identifying underserved zones with >90% accuracy |
| Network type predictability | >80% accuracy in predicting network types |
| Targeted optimization | Actionable insights for at least 3 key zones |
| Scalability | Model generalizes well to new geolocations |

#  Data Understanding – Cellular Network Performance Project

##  Dataset Overview

This dataset contains signal quality measurements from various geographic locations and cellular network types. The goal is to assess and model the quality of service across different network types and environments.

---

##  Key Features

| Column Name        | Description |
|--------------------|-------------|
| `latitude`         | Latitude coordinate where signal was measured |
| `longitude`        | Longitude coordinate where signal was measured |
| `timestamp`        | Time of measurement |
| `tool`             | Measurement tool used (e.g., BB60C, srsRAN, BladeRFxA9) |
| `network_type`     | Type of mobile network (3G, 4G, LTE, 5G) |
| `locality`         | Area classification (Urban, Suburban, Rural) |
| `signal_strength`  | Signal power in dBm |
| `signal_quality`   | Quality of the signal (percentage) |
| `data_throughput`  | Data transmission speed (Mbps) |
| `latency`          | Network delay in milliseconds (ms) |

---


# Exploratory Data Analysis (EDA)

## Uni-variate Analysis

![image](https://github.com/user-attachments/assets/447f20fc-faf5-4e9d-87b2-aae16a20becd)

![image](https://github.com/user-attachments/assets/f78f5225-f394-4684-83dc-15e5a5b6afad)

**Observations**

- **Signal Strength (dBm)**: The distribution is approximately normal, centered around -90 dBm, which represents typical signal strengths within the dataset.
- **Data Throughput (Mbps)**: This feature is highly skewed towards lower values, indicating that most measurements show lower data speeds, with fewer instances of high throughput.


![image](https://github.com/user-attachments/assets/cf8f5f53-72a7-4a42-a69f-d766933506c9)

**Observations**
1. **Proportions (Percentages)**: Each segment of the donut chart has a numerical label indicating its percentage contribution to the total.
- LTE: 25.1%
- 4G: 25.1%
- 3G: 25.0%
- 5G: 24.8%

2. **Near-Even Distribution**: A key observation is that all four network types have a very similar, almost equal, distribution. Each category accounts for approximately 25% of the total, suggesting a nearly uniform distribution among these network types.


## Bi-Variate Analysis

![image](https://github.com/user-attachments/assets/848b97c8-b647-409b-a1c5-52e1be9a6eae)

**Observations**
1. **Data Distribution** - Inverted Trapezoid/Funnel Shape: The most prominent observation is the shape of the data distribution. It forms an approximate inverted trapezoid or funnel shape.
 - At higher (less negative) Signal Strength values (e.g., from -90 to -78), the Data Throughput values are mostly concentrated at the lower end (around 0 to 10), forming a dense, narrow band.
 - As Signal Strength decreases (becomes more negative, moving to the left), the range of observed Data Throughput values widens significantly.
 - For instance, at Signal Strength around -100, Data Throughput ranges from near 0 up to 100.
 - At the far left (around -110 Signal Strength), Data Throughput again appears to be more scattered but still covers a wide range.

2. **Implied Relationship:**
 - Strong Signal Strength (less negative): Tends to correspond to very low Data Throughput. This is counter-intuitive if higher signal strength is expected to mean better performance, and might suggest some other limiting factor or a specific scenario where strong signals result in low throughput.
 - Weaker Signal Strength (more negative): Allows for a much wider range of Data Throughput values, including high throughput. This is also interesting and suggests that strong signals might not be the primary determinant of high throughput in this dataset, or that other factors become more dominant with stronger signals.
 - Dense Area: The densest concentration of points is in the bottom-right corner, where Signal Strength is higher (less negative) and Data Throughput is very low.

![image](https://github.com/user-attachments/assets/12143b9a-6091-4f16-9d00-6e557840c2c7)

**Observations**
- It appears that 3G has the least negative (strongest) signal strength among the observed network types, even though it's still very weak given the scale.
- LTE and 4G have very similar signal strengths, which are slightly weaker (more negative) than 3G.
- 5G has the most negative (weakest) signal strength among the categories.

![image](https://github.com/user-attachments/assets/630561bd-1a37-4e8d-9b17-1b62259328fb)

**Observations**
- 5G demonstrates vastly superior Data Throughput compared to all other network types. It is an order of magnitude higher than 4G, and significantly higher than 3G and LTE.
- There's a clear descending trend in Data Throughput from 5G to 4G, then to 3G, and finally to LTE.


![image](https://github.com/user-attachments/assets/90e4fd32-ab4a-4789-99e4-ff2436be45fe)

**Observations**
1. **Signal Strength Trend:** The signal strength fluctuates significantly across the various times of the day.
- Strongest Signal (Least Negative): The signal appears strongest during Mid-Morning, peaking at approximately -89.8 dBm. There's another relatively strong period during Night (around -90.0 dBm).
- Weakest Signal (Most Negative): The signal is weakest at Midnight, dropping significantly to approximately -90.3 dBm. It also shows dips during Late Evening and Early Morning.

2. **Fluctuation Patterns:** The graph shows a clear diurnal pattern, with signal strength generally improving from early morning through mid-morning, then declining towards midday and late evening, a sharp drop at midnight, followed by a recovery into the night.

3. **Range of Fluctuation:** The difference between the strongest (Mid-Morning: ~-89.8 dBm) and weakest (Midnight: ~-90.3 dBm) average signal strength is relatively small (around 0.5 dBm), suggesting that while there are consistent patterns, the absolute variation in average signal strength is not extremely large.

![image](https://github.com/user-attachments/assets/3430f3f7-da1f-4227-8707-115851f79070)

**Observations**
1. Throughput Trend: The average throughput fluctuates significantly across the different times of day, showing a distinct pattern.

2. Peaks (Highest Throughput):
- Late Evening shows the highest throughput, peaking at approximately 16.8 mbps.
- Early Afternoon and Midnight also exhibit high throughput, around 16.7 mbps.
- Early Morning is also relatively high at around 16.65 mbps.

3. Troughs (Lowest Throughput):
- The lowest throughput occurs during Night, dropping significantly to approximately 14.8 mbps.
- Late Afternoon also shows a dip to around 15.8 mbps.

4. General Pattern: The graph indicates a generally good throughput during early afternoon, early morning, late evening, and midnight. There are noticeable dips in throughput during late afternoon, and a very sharp drop during the "Night" period.

5. Range of Fluctuation: The difference between the highest (Late Evening: ~16.8 mbps) and lowest (Night: ~14.8 mbps) average throughput is approximately 2 mbps. This represents a noticeable, though not extreme, daily variation in network performance.

## Multi-variate Analysis

![image](https://github.com/user-attachments/assets/51536613-d0c7-49dd-8058-451fb2dcf14c)

**Observations**
1. Overall Data Distribution: The data points form a wide, somewhat "inverted trapezoid" or "funnel" shape, similar to a previous observation, where throughput generally has a wider range at weaker signal strengths and a narrower range at stronger signal strengths, especially for 5G.

2. Distribution by Network Type: This is the most insightful observation due to the color encoding:
- 5G (red points): These points constitute the vast majority of observations at higher Data Throughput values (above ~10). They are broadly distributed across most Signal Strength values, from approximately -108 up to -78, covering the entire range of throughputs from near 0 up to 100. This suggests that 5G connections are responsible for almost all high-throughput data.
- 3G (blue points), 4G (orange points), and LTE (green points): These network types are primarily clustered at very low Data Throughput values, typically below 10.
- They form distinct, narrow horizontal bands at the bottom of the plot.
- LTE (green) appears to be slightly higher than 3G (blue), and 4G (orange) slightly higher than LTE, but all are clearly segregated to the very low throughput region.
- These network types are also predominantly observed at stronger signal strengths (less negative, generally from around -100 to -78).

3. Implied Performance Comparison:
- 5G clearly offers superior Data Throughput performance compared to 3G, 4G, and LTE. While 5G can also have low throughput, it's the only network type observed achieving high throughput.
- Conversely, 3G, 4G, and LTE are consistently associated with very low throughput, regardless of signal strength within their observed range.

![image](https://github.com/user-attachments/assets/ef206c48-7284-4336-a0d2-343df9d9e318)

**Observations**
1. Key Correlations:

- Signal Strength vs. Latency: There is a strong positive correlation of 0.50 between signal_strength_(dbm) and latency_(ms). This is highly counter-intuitive: it suggests that as signal strength increases (becomes less negative, closer to 0), latency also increases. This is a significant finding and goes against the typical expectation that stronger signals lead to lower latency.
- Signal Strength vs. Data Throughput: There is a strong negative correlation of -0.48 between signal_strength_(dbm) and data_throughput_(mbps). This is also counter-intuitive: it suggests that as signal strength increases, data throughput decreases. This aligns with the previous scatter plot observation where higher throughput was observed at weaker signal strengths for 5G.
- Data Throughput vs. Latency: There is a strong negative correlation of -0.67 between data_throughput_(mbps) and latency_(ms). This is an expected and logical correlation: as data throughput increases, latency decreases. This is a desirable relationship.

2. Measurement Variables:

- bb60c_measurement_(dbm), srsran_measurement_(dbm), and bladerfxa9_measurement_(dbm) are very highly positively correlated with signal_strength_(dbm) (0.62, 0.62, 0.62 respectively). This suggests these are likely alternative or related measurements of signal strength.
- These measurement variables also have moderate positive correlations with latency_(ms) (0.52, 0.52, 0.52) and moderate negative correlations with data_throughput_(mbps) (-0.36, -0.36, -0.36), consistent with their strong correlation to signal_strength_(dbm).

3. Latitude and Longitude:

- latitude and longitude show very weak or negligible correlations (close to 0.00 or -0.01) with all other network performance metrics (signal_strength, data_throughput, latency, and the measurement variables). This suggests that geographical location within the observed area does not have a strong linear relationship with these network performance indicators.

# Modeling

## Regression

Here we will begin the regression modeling for signal strength and throughput
The major models include
1. Linear regressor
2. Random Forest regressor
3. XGBoost Regressor

## Signal Strength Regression
### Best Model

Tuned XGBoost Performance:
R² Score: 0.8926
MAE: 1.2052
MSE: 3.0366
RMSE: 1.7426

![image](https://github.com/user-attachments/assets/c26d7cd2-845d-4e57-8e08-224079cf1447)

**Observations**

1. Excellent Fit

- Points cluster tightly around the 45° line, confirming the high R² (0.8926).

- Nearly identical to the Random Forest plot, reflecting their statistically equivalent performance.

2. Minor Deviations

- Underprediction (Actual ≈ -80 to -90): Points slightly below the line (model conservative for mid-range values).

- Overprediction (Actual ≈ -100 to -110): Points slightly above the line (model over-optimistic for extreme lows).

3. Consistent Error Spread

- No extreme outliers; errors are uniformly distributed (low RMSE = 1.7426).

- Slightly tighter clustering than default XGBoost, showing tuning benefits.


## Throughput Prediction

### Best model

XGBoost for Throughput Prediction
R² Score: 0.9995
MAE: 0.0259
MSE: 0.0067
RMSE: 0.0820

![image](https://github.com/user-attachments/assets/07abc0d8-2dc0-4798-85dd-846c19f58307)

**Observations**

1. Near-Perfect Alignment

- Points closely follow the 45° diagonal, confirming the model’s exceptional accuracy (R² ≈ 0.9995).

- No visible bias—predictions are equally reliable across low (0–5 Mbps) and high (15–20 Mbps) throughput ranges.

2. Micro-Deviation Analysis

- Tiny scatter (barely visible) explains the ultra-low MAE (0.0259) and RMSE (0.0820).

- Any minor errors are uniformly distributed, with no systematic over/under-prediction.

3. Comparison to Initial Model

- Before optimization: Points were scattered, especially at high throughput.

- After optimization: Achieved laboratory-grade precision due to:

- Log transformation (handled extreme values)

- Outlier removal (reduced noise)

## Classification

Random Forest
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       842
           1       1.00      1.00      1.00       823
           2       1.00      1.00      1.00       853
           3       1.00      1.00      1.00       848

    accuracy                           1.00      3366
   macro avg       1.00      1.00      1.00      3366
weighted avg       1.00      1.00      1.00      3366

### Checking for overfitting 

Training Performance:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      3366
           1       1.00      1.00      1.00      3396
           2       1.00      1.00      1.00      3325
           3       1.00      1.00      1.00      3376

    accuracy                           1.00     13463
   macro avg       1.00      1.00      1.00     13463
weighted avg       1.00      1.00      1.00     13463


Test Performance:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       842
           1       1.00      1.00      1.00       823
           2       1.00      1.00      1.00       853
           3       1.00      1.00      1.00       848

    accuracy                           1.00      3366
   macro avg       1.00      1.00      1.00      3366
weighted avg       1.00      1.00      1.00      3366

**Insights**

- Trains perfectly

- Generalizes perfectly to test data

- Scores perfectly across 5-fold cross-validation

No overfitting,  based on this train/test split and cross-validation. This suggests:

- There is likely very strong signal in the data.

- The target is likely easily separable with the current features.

There is no apparent data leakage or overfitting.

![image](https://github.com/user-attachments/assets/42737f0e-313d-48ff-8569-ef6e0881c159)

**Observations and Insights**

1. High True Positives:
- LTE: 842 instances were correctly classified as LTE.
- 4G: 823 instances were correctly classified as 4G.
- 3G: 853 instances were correctly classified as 3G.
- 5G: 848 instances were correctly classified as 5G.

2. Zero Off-Diagonal Values: All the off-diagonal elements in this confusion matrix are 0. This is a very significant observation.

In conclusion, the confusion matrix shows a model that achieved perfect classification across all four classes (LTE, 4G, 3G, 5G). 

# Conclusions

1. Balanced Network Distribution:

- The dataset showed a nearly equal distribution across LTE, 4G, 3G, and 5G networks (~25% each), indicating that the dataset is well-suited for comparative performance analysis across network types.

2. Strongest Signal in 3G, Weakest in 5G:

- 3G networks had the highest signal strength, likely due to their use of lower-frequency bands which travel farther and penetrate buildings better. Conversely, 5G often operates on higher-frequency bands (e.g., mmWave), which have weaker propagation characteristics.

3. Superior Throughput in 5G:

- Despite its weaker signal strength, 5G provided the highest data throughput, showcasing its advanced technology (e.g., massive MIMO, wide bandwidth) designed for high-speed data transmission. Older technologies like 3G and LTE showed lower throughput, aligned with their limited capacity.

4. Latency Trends:

- LTE and 3G recorded the highest latency, consistent with their older architecture. In contrast, 4G and 5G networks had the lowest latency, with 5G designed specifically for ultra-reliable low-latency communications (URLLC).

5. Temporal Pattern in Signal Strength:

- Signal strength peaked during mid-morning hours, possibly due to lower congestion and favorable atmospheric conditions. It was lowest at midnight, potentially due to signal degradation or tower maintenance.

6. Lowest Latency in Late Evening:

- Latency was minimal in the late evening, which may reflect decreased network usage, leading to faster response times. The highest latency occurred late at night, possibly due to scheduled network operations or maintenance.

7. Throughput Peaks at Night:

- Data throughput was highest in the late evening and midnight hours, possibly due to reduced user load. The lowest throughput occurred during nighttime hours when users may consume high-bandwidth content, causing temporary network congestion.

8. XGBoost for Signal Strength Prediction:

- The XGBoost regressor effectively captured complex non-linear relationships in signal strength data, achieving strong performance (R² = 0.89, MAE = 1.21, RMSE = 1.74), making it a reliable model for real-world applications.

9. Exceptional Throughput Prediction Performance:

- After outlier removal and log transformation, XGBoost achieved near-perfect performance (R² = 0.99, MAE = 0.03, RMSE = 0.08), suggesting the presence of strong, clean patterns in the throughput data with minimal noise or bias.

10. Perfect Classification with Random Forest:

- The Random Forest Classifier achieved 100% accuracy and F1-score for zone and network type classification, indicating well-separated classes and effective feature representation, with no signs of overfitting or data leakage.

# Recommendations

1. Enhance 5G Signal Strength:

- Invest in denser 5G infrastructure (e.g., small cells) to address weaker signal strength issues and ensure coverage parity with 3G and LTE.

2. Maintain 3G and LTE Support in Rural Areas:

- Given their strong signal characteristics, 3G and LTE networks can continue serving areas where 5G infrastructure is sparse or signal penetration is critical.

3. Optimize for Peak Throughput Periods:

- Network resources can be dynamically allocated to match high-throughput periods (e.g., late evening) to maintain service quality.

4. Schedule Maintenance Strategically:

- Since late night hours show the highest latency, network maintenance activities should be optimized to reduce performance degradation during these periods.

5. Use XGBoost in Production Systems:

- Due to its high accuracy and generalizability, XGBoost should be integrated into real-time systems for signal strength and throughput forecasting.

6. Deploy Ensemble Models for Classification Tasks:

- The Random Forest Classifier’s exceptional results make it ideal for deployment in zone or network classification systems, especially where interpretability and robustness are key.

7. Monitor Network Usage Trends Over Time:

- Continue tracking temporal patterns in signal and throughput to anticipate demand and preemptively manage resources.

# Deploying the Models using Streamlit

## Signal Strength Prediction

![app - Google Chrome 6_16_2025 3_55_08 PM](https://github.com/user-attachments/assets/9a3e5f69-a35e-44f7-b539-bdedfa050ec9)



![app - Google Chrome 6_16_2025 3_55_27 PM](https://github.com/user-attachments/assets/face742f-72a7-4286-ae29-3f98b8190889)

## Throughout Prediction

![app - Google Chrome 6_16_2025 4_05_23 PM](https://github.com/user-attachments/assets/18fe8ff9-8288-4aa0-99bd-fa6897cefdfa)


![throughput - Google Chrome 6_16_2025 4_05_44 PM](https://github.com/user-attachments/assets/ef1717fa-60d4-475d-b638-b9dc7de18e9c)


## Network Type Prediction

![network - Google Chrome 6_16_2025 4_17_43 PM](https://github.com/user-attachments/assets/2d84d372-efc3-44e4-be40-69297197e481)


![network - Google Chrome 6_16_2025 4_18_00 PM](https://github.com/user-attachments/assets/efbbc45e-962a-4711-bb55-f4d295df2c7e)




