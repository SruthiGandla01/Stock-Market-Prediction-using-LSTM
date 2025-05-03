# 📈 Stock Market Prediction using Deep Learning (LSTM vs CNN)

This project focuses on predicting the **next-day stock market direction** using deep learning models. We developed and compared **Convolutional Neural Networks (CNN)** and **Long Short-Term Memory (LSTM)** networks on a rich dataset containing technical, economic, and global market indicators.

---

## 🔍 Project Overview

The objective is to forecast whether the stock market (S&P 500, NASDAQ, NYSE) will move **up** or **down** the next day, based on historical trends and macroeconomic signals. This project helps illustrate how deep learning can be applied to financial time series for smarter investment insights.

---

## 📊 Dataset

- **Source:** UCI Machine Learning Repository  
- **Timeframe:** 2010 – 2017  
- **Indices Covered:** S&P 500, NASDAQ, NYSE  
- **Features (82 total):**
  - Technical indicators (e.g., moving averages)
  - Economic variables (e.g., interest rates, oil prices)
  - Global indices, currency, futures, commodities

---

## ⚙️ Data Preprocessing

- Missing value handling using **forward-fill** and **backward-fill**
- **MinMaxScaler** for feature normalization
- **30-day sliding windows** for sequence creation
- **SMOTE + undersampling** to address class imbalance
- Binary target creation:  
  `Target = 1` if next-day return > 0.5%, else `0`
- Chronological **train/test split** (80/20)

---

## 🧠 Models

### 📌 CNN Model
- Conv1D layers for short-term feature extraction
- **Residual connection** with `Add()` layer
- BatchNormalization + Dropout for regularization
- GlobalMaxPooling + Dense classification
- **Loss Function:** Focal Loss  
- **Output:** Sigmoid activation

### 📌 LSTM Model
- 2 stacked LSTM layers for sequential modeling
- BatchNormalization + Dropout for generalization
- Dense output with sigmoid
- **Loss Function:** Focal Loss

---

## 🛠️ Training & Evaluation

- Optimizer: **Adam**
- Epochs: 60  
- Batch Size: 32  
- Callbacks: EarlyStopping, ReduceLROnPlateau  
- Threshold tuning using **Precision-Recall curve**  
- Evaluation metrics: Accuracy, Precision, Recall, F1-Score

---

## 📈 Results

| Model | Accuracy | Class 0 F1 | Class 1 F1 | Macro F1 |
|-------|----------|------------|------------|----------|
| CNN   | 54.99%   | 66%        | 72%        | 49%      |
| LSTM  | 73.01%   | 87%        | 83%        | 60%      |

✅ **LSTM outperformed CNN**, demonstrating better capability in capturing long-term dependencies in stock price trends.

---

## 💡 Conclusion

- CNN performs well on short-term patterns but is limited for sequential forecasting.
- LSTM effectively models long-term dependencies and generalizes better across market conditions.
- Future enhancements: hybrid **CNN-LSTM model**, external **news sentiment**, or **Transformer-based architectures**.

---

## 🧰 Tech Stack

- Python, TensorFlow, Keras, Scikit-learn, Pandas, Matplotlib
- Focal Loss, SMOTE, MinMaxScaler

---

## 📎 Contributors

- Sruthi Gandla - gandla.s@northeastern.edu
- Aishwariya Alagesan - alagesan.a@northeastern.edu  


---

