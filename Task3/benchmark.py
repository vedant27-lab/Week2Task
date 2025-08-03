import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from timeseries_utils import rolling_stats_numpy, rolling_stats_pandas, ewma_numpy, ewma_pandas

def benchmark_function(func, *args):
    start = time.perf_counter()
    func(*args)
    end = time.perf_counter()
    return end - start

if __name__ == "__main__":
    N = 1_000_000
    window = 100
    span = 50
    alpha = 2/(span+1)

    print("Generating synthetic data...")
    data_np = np.random.randn(N)
    data_pd = pd.Series(data_np)

    results = []
    
    print("Benchmarking rolling statistics...")

    t1 = benchmark_function(rolling_stats_numpy, data_np, window)
    results.append({"Function": "rolling_stats_numpy", "time_sec": t1})

    t2 = benchmark_function(rolling_stats_pandas, data_pd, window)
    results.append({"Function": "rolling_stats_pandas", "time_sec": t2})

    print("Benchmarking EWMA...")

    t3 = benchmark_function(ewma_numpy, data_np, alpha)
    results.append({"function": "ewma_numpy", "time_sec": t3})

    t4 = benchmark_function(ewma_pandas, data_pd, span)
    results.append({"function": "ewma_pandas", "time_sec": t4})

    df = pd.DataFrame(results)
    df.to_csv("final_results.csv", index=False)
    print("\nBenchmark results saved to final_results.csv:\n")
    print(df)

    #ploting the results
    df.plot(kind="bar", x="Function", y="time_sec", legend=False, title="Runtime Computations:")
    plt.ylabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig("Benchmark_chart.png")
    plt.show()