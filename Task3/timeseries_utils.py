import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def rolling_stats_pandas(series: pd.Series, window: int) -> pd.DataFrame:
    """
    Computes rolling mean and variance using pandas.

    Parameters:
        series (pd.Series): Input time-series data
        window (int): Rolling window size

    Returns:
        pd.DataFrame: with 'mean' and 'variance' columns
    """

    rolling = series.rolling(window=window)
    return pd.DataFrame({
        "mean": rolling.mean(),
        "variance": rolling.var()
    })

def rolling_stats_numpy(array: np.ndarray, window: int) -> tuple:
    """
    Computes rolling mean and variance using NumPy (vectorized approach).

    Paramenters:
        array (np.ndarray): 1D time-series array
        window (int): Rolling window size

    Returns:
        (mean_array, var_array): Tuple of NumPy arrays
    """
    if array.ndim != 1:
        raise ValueError("Only 1D arrays are supported.")
    
    cumsum = np.cumsum(np.insert(array, 0, 0))
    cumsum_sq = np.cumsum(np.insert(array ** 2, 0, 0))

    sum_ = cumsum[window:] - cumsum[:-window]
    sum_sq = cumsum_sq[window:] - cumsum_sq[:-window]

    mean = sum_ / window
    var = (sum_sq / window) - (mean ** 2)
    
    return mean, var

def ewma_pandas(series: pd.Series, span: int) -> pd.Series:
    """
    Compute Exponentially Weighted Moving Average using pandas.

    Parameters:
        series (pd.Series): Input data
        span (int): span for smoothing

    Returns:
        pd.series: EWMA result
    """
    return series.ewm(span=span, adjust=False).mean()

def ewma_numpy(array: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute EWMA manually using NumPy.

    Parameters:
        array (np.ndarray): 1D array of data
        alpha (float): smoothing factor (0 < alpha <= 1)

    Returns:
        np.ndarray: EWMA result
    """
    ewma = np.zeros_like(array)
    ewma[0] = array[0]
    for t in range(1, len(array)):
        ewma[t] = alpha * array[t] + (1 - alpha) * ewma[t-1]
    return ewma

def fft_bandpass_filter(signal: np.ndarray, low: float, high: float, sample_rate: float) -> np.ndarray:
    """
    Bandpass filter using FFT

    Parameters:
        signal (np.ndarray): 1D input signal
        low (float): low cutoff frequency in Hz
        high (float): high cutoff frequency in Hz
        sample_rate (float): sampling rate in Hz

    Returns:
        np.ndarray: Filtered signal (real-valued)
    """

    n = len(signal)
    fft_vals = np.fft.fft(signal)
    freqs = np.fft.fftfreq(n, d=1/sample_rate)

    #Creating mask
    mask = (np.abs(freqs) >= low) & (np.abs(freqs) <= high)
    #utilizing mask
    filtered_fft = fft_vals*mask

    filtered_signal = np.fft.ifft(filtered_fft).real
    return filtered_signal



if __name__ == "__main__":
    x = np.random.randn(1000)
    series = pd.Series(x)
    result_pandas = rolling_stats_pandas(series, 20)
    print("Pandas Rolling Stats:")
    print(result_pandas.dropna().head())

    mean_np, var_np = rolling_stats_numpy(series.values, 20)
    print("\nNumPy Rolling Stats:")
    print("Mean:", mean_np[:5])
    print("Variance:", var_np[:5])

    alpha = 2/(11)

    ew_np = ewma_numpy(x, alpha)
    ew_pd = ewma_pandas(series, span=10)

    print(ew_np[:5])
    print(ew_pd.head())

    t = np.linspace(0, 1, 1000, endpoint=False)
    signal = np.sin(2*np.pi*3*t) + 0.5*np.random.randn(len(t))

    filtered = fft_bandpass_filter(signal, low=2, high=4, sample_rate=1000)

    plt.figure(figsize=(10,4))
    plt.plot(t, signal, label="Original")
    plt.plot(t, filtered, label="Filtered", linewidth=2)
    plt.title("FFT Bandpass Filter")
    plt.legend()
    plt.grid(True)
    plt.show()

    

    
