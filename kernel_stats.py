import sys
import pandas as pd
import numpy as np

def main(csv_file):
    # Read CSV
    df = pd.read_csv(csv_file)

    # Order by Kernel Start (ns)
    df = df.sort_values(by="Kernel Start (ns)").reset_index(drop=True)

    # Calculate end times for each kernel
    df["Kernel End (ns)"] = df["Kernel Start (ns)"] + df["Kernel Dur (ns)"]

    # Calculate gaps (difference between current start and previous end)
    gap_times = df["Kernel Start (ns)"].iloc[1:].values - df["Kernel End (ns)"].iloc[:-1].values

    # Total execution time (from first start to last end)
    total_time = df["Kernel End (ns)"].iloc[-1] - df["Kernel Start (ns)"].iloc[0]

    # Sum kernel times
    kernel_time = df["Kernel Dur (ns)"].sum()

    # Sum gap times
    gap_time = gap_times.sum()

    # Percentages
    kernel_pct = kernel_time / total_time * 100
    gap_pct = gap_time / total_time * 100

    print("=== Percentage Breakdown ===")
    print(f"Kernel Time: {kernel_time} ns ({kernel_pct:.2f}%)")
    print(f"Gap Time:    {gap_time} ns ({gap_pct:.2f}%)")
    print(f"Total Time:  {total_time} ns, i.e. {total_time / 1e6:.2f} ms\n")

    # Kernel stats
    print("=== Kernel Statistics (ns) ===")
    print(f"Mean: {df['Kernel Dur (ns)'].mean():.2f}")
    print(f"Std:  {df['Kernel Dur (ns)'].std():.2f}")
    print(f"Min:  {df['Kernel Dur (ns)'].min()}")
    print(f"Max:  {df['Kernel Dur (ns)'].max()}")
    # print the number of kernels
    print(f"Number of Kernels: {df.shape[0]}\n")

    # Gap stats
    print("=== Gap Statistics (ns) ===")
    print(f"Mean: {np.mean(gap_times):.2f}")
    print(f"Std:  {np.std(gap_times):.2f}")
    print(f"Min:  {np.min(gap_times)}")
    print(f"Max:  {np.max(gap_times)}")
    # print the number of gaps
    print(f"Number of Gaps: {gap_times.shape[0]}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_kernels.py <file.csv>")
        sys.exit(1)
    main(sys.argv[1])