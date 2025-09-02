# fx-graph-research

To run nsys
```bash
make clean
make mod
make env
make run-original # with default modeling file
make run-fixed

for f in kernel_stats*.log; do
  echo "===== $f ====="
  cat "$f"
done
```

Example output
```
===== kernel_stats_fixed_20250902_005654.log =====
=== Percentage Breakdown ===
Kernel Time: 646504874 ns (34.12%)
Gap Time:    1248263229 ns (65.88%)
Total Time:  1894768103 ns, i.e. 1894.77 ms

=== Kernel Statistics (ns) ===
Mean: 8935.67
Std:  42253.09
Min:  960
Max:  1387473
Number of Kernels: 72351

=== Gap Statistics (ns) ===
Mean: 17253.12
Std:  17632.72
Min:  576
Max:  1559060
Number of Gaps: 72350
===== kernel_stats_original_20250902_005610.log =====
=== Percentage Breakdown ===
Kernel Time: 647169711 ns (34.45%)
Gap Time:    1231478936 ns (65.55%)
Total Time:  1878648647 ns, i.e. 1878.65 ms

=== Kernel Statistics (ns) ===
Mean: 8951.05
Std:  42284.86
Min:  928
Max:  1388111
Number of Kernels: 72301

=== Gap Statistics (ns) ===
Mean: 17032.90
Std:  16926.62
Min:  576
Max:  950282
Number of Gaps: 72300
```

Is torch.compile really enabled??