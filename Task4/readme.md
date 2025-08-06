# 🧠 Delhi AQI Prediction using Machine Learning (Jupyter Version)

This project predicts Delhi's Air Quality Index (AQI) using historical pollutant data and machine learning models developed inside Jupyter Notebook.

---

## 📊 Dataset Overview

- **Source**: Delhi Air Quality Dataset (Kaggle)
- **Size**: ~1000 rows
- **Columns**:
  - Pollutants: `PM2.5`, `PM10`, `NO2`, `SO2`, `CO`, `Ozone`
  - Time: `Date`, `Month`, `Year`, `Days`, `Holidays_Count`
  - Target: `AQI`

---

## 🧪 Feature Engineering

- Removed nulls and duplicate entries
- Created useful ratios like:
  - `PM_ratio = PM2.5 / PM10`
  - `NO2_SO2_ratio = NO2 / SO2`
- Normalized/Standardized values
- Selected numerical features based on correlation

---

## 🤖 Models Trained

| Model              | RMSE   | R² Score |
|-------------------|--------|----------|
| Linear Regression | 38.10  | 0.89     |
| Random Forest      | 29.13  | 0.93     |
| XGBoost            | 28.87  | 0.94     |

✅ **XGBoost showed the best performance** in terms of error and fit.

---

## 📈 Visual Comparison

### RMSE Comparison

![RMSE](rmse_comparison.png)

### R² Score Comparison

![R²](r2_comparison.png)

---

## 📦 Project Files

├── model.ipynb # Main notebook with model training
├── cleanedData.csv # Cleaned version of the dataset
├── rmse_comparison.png
└── r2_comparison.png
├── README.md
└── requirements.txt


---

## 💡 Future Work

- Integrate with Streamlit for AQI prediction
- Deploy model to web app
- Use hyperparameter tuning for even better accuracy

---

## 👨‍💻 Author

**Vedant Patil**  
Computer Engineering, SPPU  
Deep Learning Intern @ CyArt

---

