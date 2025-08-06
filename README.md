# 🚀 Week 2 – Machine Learning Projects by Vedant Patil

This repository contains my Week 2 internship work at **CyArt**, covering 4 individual tasks related to machine learning, deep learning foundations, time series processing, and model-based predictions.

Each task was built **step-by-step from scratch** and demonstrates both my technical learning and practical application.

---

## 📁 Folder Structure

Week2-Tasks/
├── Task1/
│ ├── train.ipynb
│ ├── model.py
│ ├── Loss Curve.png
│ ├── Model Output.png
│ └── Readme.md
│
├── Task2/
│ ├── app.py
│ ├── sample.txt
│ ├── nlp_pipeline.py
│ ├── test_requests.http
│ └── readme.md
│
├── Task3/
│ ├── benchmark.py
│ ├── final_results.csv
│ ├── Filter_1.png
│ ├── Benchmark_chart.png
│ ├── timeseries_utils.py
│ └── readme.md
│
├── Task4/
│ ├── cleanedData.csv
│ ├── data_cleaning.ipynb
│ ├── final_dataset.csv
│ ├── model_r2_comparison.png
│ ├── model_rmse_comparison.png
│ ├── model.ipynb
│ └── readme.md


---

## 🔹 Task 1: Neural Network from Scratch (NumPy)

### 📌 Objective:
Build a simple feedforward neural network from scratch using NumPy — no external ML libraries.

### 🛠️ Components:
- `Layer_Dense`: Implements weights, biases, forward and backward pass
- Activations:
  - `ReLU`, `Sigmoid`
- Loss:
  - `MeanSquaredError`
- Optimizer:
  - `Stochastic Gradient Descent (SGD)`

### 📈 Output:
- Training loop runs with visible loss reduction
- Layer-wise backpropagation validated
- PNG graphs included: `Loss Curve`, `Model Output`

### 📁 Files:
- `train.ipynb`: Full training loop
- `model.py`: Modularized NN code
- `Loss Curve.png`: Epoch loss visualization
- `Model Output.png`: Final output

---

## 🔹 Task 2: Sentiment Analysis API using Flask

### 📌 Objective:
Create a REST API that performs sentiment classification (Positive, Negative, Neutral) on a given sentence.

### 🛠️ Stack Used:
- `Flask` for web API
- `TextBlob` for sentiment polarity
- `Postman` for testing endpoints

### ⚙️ Endpoint:

- **POST /analyze**
```json
Request:
{
  "text": "The air quality is really bad."

Response:
{
  "sentiment": "negative"
}

📁 Files:
* app.py: Main Flask app
* nlp_pipeline.py: Sentiment logic
* sample.txt: For CLI input testing
* test_requests.http: Ready-to-use tests

ask 3: Rolling Mean & Variance – Pandas vs NumPy
📌 Objective:
Compare rolling window statistics (mean/variance) between Pandas and NumPy.

🛠️ Functions:
rolling_stats_pandas(): Uses pandas.Series.rolling

rolling_stats_numpy(): Vectorized NumPy with cumsum

🔍 Insights:
NumPy version is much faster for large arrays

Results are validated to be nearly identical

📊 Outputs:
Benchmark_chart.png: Pandas vs NumPy timing chart

Filter_1.png: Visual rolling stats

final_results.csv: Benchmarked results

📁 Files:
benchmark.py: Speed test and visualization

timeseries_utils.py: Both function definitions

readme.md: Mini README inside Task3 folder

🔹 Task 4: AQI Prediction with ML (Delhi Dataset)
📌 Objective:
Train multiple ML models to predict the Air Quality Index (AQI) for Delhi using a public Kaggle dataset.

📊 Dataset Columns:
Features: PM2.5, PM10, NO2, SO2, CO, Ozone

Date context: Month, Year, Days, Holidays_Count

Target: AQI

🔬 Preprocessing:
Null value handling

Added feature ratios: PM_ratio, NO2_SO2_ratio

Feature scaling for linear models
| Model             | RMSE  | R² Score |
| ----------------- | ----- | -------- |
| Linear Regression | 42.30 | 0.72     |
| Random Forest     | 30.70 | 0.85     |
| XGBoost           | 27.90 | 0.89     |

🏆 Best Performer:
XGBoost with lowest RMSE and highest R².

📈 Graphs:
rmse_comparison.png: RMSE Bar Chart
r2_comparison.png: R² Bar Chart

📁 Files:
model.ipynb: Full pipeline in Jupyter
cleanedData.csv: Cleaned dataset used

Requirements
All tasks are based on:
Python 3.10+
NumPy, Pandas, Matplotlib, Scikit-learn, TextBlob
Flask (Task 2), XGBoost (Task 4)
You can install all dependencies using:

pip install -r requirements.txt

✍️ Author
Vedant Patil
Deep Learning Intern – CyArt
Computer Engineering @ SPPU

