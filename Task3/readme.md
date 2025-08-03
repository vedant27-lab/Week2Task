# Task 3: High-Performance Time Series Transformation with NumPy & pandas

This task focuses on building fast, scalable time-series utilities to process datasets with over **1 million rows**, comparing performance between NumPy and pandas implementations.

It covers:
- Rolling statistics (mean, variance)
- Exponentially Weighted Moving Average (EWMA)
- FFT-based spectral filtering
- Performance benchmarking

---

## 🚀 Features

✅ Efficient rolling mean and variance (NumPy & pandas)  
✅ EWMA implementation using vectorized NumPy + pandas  
✅ FFT-based bandpass filter  
✅ Benchmark runtime comparisons  
✅ Results exported to CSV + optional visualization  

---

## 📁 File Structure

| File | Description |
|------|-------------|
| `timeseries_utils.py` | Contains all time-series processing functions |
| `benchmark.py`        | Runs timed comparisons, saves results to CSV |
| `results.csv`         | Benchmark runtime results |
| `benchmark_chart.png` | Optional runtime comparison chart |
| `README.md`           | This documentation file |

---

## ⚙️ How to Run

### 1. Install Dependencies
```bash
pip install numpy pandas matplotlib

2. Run Benchmark Script
python benchmark.py

This will:

* Generate 1 million rows of synthetic time-series data
* Time each function
* Save results to final_results.csv
* Show comparison chart (if matplotlib installed)

Functions Implemented
Rolling Mean & Variance

* rolling_stats_numpy(data, window)
* rolling_stats_pandas(data, window)

EWMA (Exponentially Weighted Moving Average)

* ewma_numpy(data, alpha)
* ewma_pandas(data, span)

FFT-Based Bandpass Filter:
fft_bandpass_filter(data, low_freq, high_freq, sample_rate)

Sample Benchmark Output

Function	            Time (sec)
rolling_stats_numpy	    0.134
rolling_stats_pandas	0.397
ewma_numpy	            0.081
ewma_pandas	            0.276

(Actual times will vary by system.)

📚 References
NumPy Docs

pandas Rolling Windows

FFT Tutorial (SciPy)


















