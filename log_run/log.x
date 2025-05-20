[aiter] WARNING: NUMA balancing is enabled, which may cause errors. It is recommended to disable NUMA balancing by running "sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'" for more details: https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html#disable-numa-auto-balancing
[aiter] type hints mismatch, override to --> wvSpltK(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor, arg3: int, arg4: int) -> None
[W530 13:07:51.329318993 collection.cpp:1100] Warning: ROCTracer produced duplicate flow start: 1 (function operator())
[aiter] type hints mismatch, override to --> wv_splitk_small_fp16_bf16(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor, arg3: int, arg4: int) -> None
[aiter] [perf] dim: (4, 32, 8192)        dtype: torch.float16, torch avg: 16.19    us, B avg: 6.30     us, uplift: 156.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 32, 8192)        dtype: torch.bfloat16, torch avg: 30.01    us, B avg: 9.02     us, uplift: 232.8%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 32, 9216)        dtype: torch.float16, torch avg: 31.70    us, B avg: 7.13     us, uplift: 344.8%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 32, 9216)        dtype: torch.bfloat16, torch avg: 134.00   us, B avg: 10.04    us, uplift: 1234.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
max_cu_count=80
len(boundary_mnk_list)=37
Total valid (m, n, k) tuples with k divisible by 8: 686080
len(mnk_list)=138
[aiter] using soltype=0, solidx=0 for m=1 n=1 k=8 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (1, 1, 8)            dtype: torch.float16, torch avg: 7.09     us, B avg: 1.66     us, uplift: 326.7%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=1 n=1 k=8 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (1, 1, 8)            dtype: torch.bfloat16, torch avg: 7.20     us, B avg: 1.77     us, uplift: 307.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=1 n=1 k=9216 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (1, 1, 9216)         dtype: torch.float16, torch avg: 10.88    us, B avg: 4.87     us, uplift: 123.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=1 n=1 k=9216 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (1, 1, 9216)         dtype: torch.bfloat16, torch avg: 10.91    us, B avg: 5.68     us, uplift: 92.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 160, 8)          dtype: torch.float16, torch avg: 2.90     us, B avg: 3.13     us, uplift: -7.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 160, 8)          dtype: torch.bfloat16, torch avg: 3.12     us, B avg: 3.28     us, uplift: -4.7%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 160, 9216)       dtype: torch.float16, torch avg: 8.66     us, B avg: 9.17     us, uplift: -5.6%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 160, 9216)       dtype: torch.bfloat16, torch avg: 10.13    us, B avg: 10.77    us, uplift: -5.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=2 n=1 k=8 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (2, 1, 8)            dtype: torch.float16, torch avg: 7.07     us, B avg: 1.72     us, uplift: 311.0%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=2 n=1 k=8 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (2, 1, 8)            dtype: torch.bfloat16, torch avg: 7.17     us, B avg: 2.01     us, uplift: 256.7%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=2 n=1 k=9216 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (2, 1, 9216)         dtype: torch.float16, torch avg: 10.86    us, B avg: 5.34     us, uplift: 103.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=2 n=1 k=9216 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (2, 1, 9216)         dtype: torch.bfloat16, torch avg: 10.89    us, B avg: 7.31     us, uplift: 48.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 80, 8)           dtype: torch.float16, torch avg: 3.09     us, B avg: 2.48     us, uplift: 24.6%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 80, 8)           dtype: torch.bfloat16, torch avg: 3.57     us, B avg: 2.75     us, uplift: 29.8%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 80, 9216)        dtype: torch.float16, torch avg: 9.57     us, B avg: 5.76     us, uplift: 66.0%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 80, 9216)        dtype: torch.bfloat16, torch avg: 14.14    us, B avg: 7.75     us, uplift: 82.5%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=4 n=1 k=8 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (4, 1, 8)            dtype: torch.float16, torch avg: 7.05     us, B avg: 1.96     us, uplift: 259.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=4 n=1 k=8 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (4, 1, 8)            dtype: torch.bfloat16, torch avg: 7.17     us, B avg: 2.31     us, uplift: 211.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=4 n=1 k=9216 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (4, 1, 9216)         dtype: torch.float16, torch avg: 10.91    us, B avg: 6.79     us, uplift: 60.6%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=4 n=1 k=9216 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (4, 1, 9216)         dtype: torch.bfloat16, torch avg: 10.86    us, B avg: 10.18    us, uplift: 6.7% [checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 80, 8)           dtype: torch.float16, torch avg: 4.22     us, B avg: 2.68     us, uplift: 57.3%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 80, 8)           dtype: torch.bfloat16, torch avg: 5.99     us, B avg: 3.06     us, uplift: 95.3%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 80, 9216)        dtype: torch.float16, torch avg: 32.66    us, B avg: 7.17     us, uplift: 355.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 80, 9216)        dtype: torch.bfloat16, torch avg: 147.08   us, B avg: 10.21    us, uplift: 1341.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 79, 9216)        dtype: torch.float16, torch avg: 32.80    us, B avg: 7.16     us, uplift: 358.0%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 79, 9216)        dtype: torch.bfloat16, torch avg: 146.99   us, B avg: 10.24    us, uplift: 1334.8%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=5 n=1 k=8 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (5, 1, 8)            dtype: torch.float16, torch avg: 7.07     us, B avg: 2.06     us, uplift: 242.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=5 n=1 k=8 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (5, 1, 8)            dtype: torch.bfloat16, torch avg: 7.20     us, B avg: 2.48     us, uplift: 190.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=5 n=1 k=5120 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (5, 1, 5120)         dtype: torch.float16, torch avg: 10.18    us, B avg: 4.74     us, uplift: 114.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=5 n=1 k=5120 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (5, 1, 5120)         dtype: torch.bfloat16, torch avg: 10.09    us, B avg: 6.97     us, uplift: 44.7%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=5 n=80 k=8 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (5, 80, 8)           dtype: torch.float16, torch avg: 9.22     us, B avg: 2.76     us, uplift: 233.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=5 n=80 k=8 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (5, 80, 8)           dtype: torch.bfloat16, torch avg: 9.24     us, B avg: 3.23     us, uplift: 186.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=5 n=80 k=5120 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (5, 80, 5120)        dtype: torch.float16, torch avg: 10.33    us, B avg: 5.31     us, uplift: 94.6%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=5 n=80 k=5120 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (5, 80, 5120)        dtype: torch.bfloat16, torch avg: 10.20    us, B avg: 7.55     us, uplift: 35.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=8 n=1 k=8 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (8, 1, 8)            dtype: torch.float16, torch avg: 7.06     us, B avg: 2.26     us, uplift: 211.8%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=8 n=1 k=8 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (8, 1, 8)            dtype: torch.bfloat16, torch avg: 7.18     us, B avg: 3.01     us, uplift: 138.5%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=8 n=1 k=5120 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (8, 1, 5120)         dtype: torch.float16, torch avg: 10.18    us, B avg: 6.00     us, uplift: 69.7%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=8 n=1 k=5120 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (8, 1, 5120)         dtype: torch.bfloat16, torch avg: 10.17    us, B avg: 9.39     us, uplift: 8.4% [checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=8 n=80 k=8 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (8, 80, 8)           dtype: torch.float16, torch avg: 9.21     us, B avg: 2.95     us, uplift: 211.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=8 n=80 k=8 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (8, 80, 8)           dtype: torch.bfloat16, torch avg: 9.25     us, B avg: 3.66     us, uplift: 153.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=8 n=80 k=5120 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (8, 80, 5120)        dtype: torch.float16, torch avg: 10.28    us, B avg: 6.57     us, uplift: 56.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=8 n=80 k=5120 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (8, 80, 5120)        dtype: torch.bfloat16, torch avg: 10.20    us, B avg: 9.90     us, uplift: 3.0% [checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=8 n=79 k=5120 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (8, 79, 5120)        dtype: torch.float16, torch avg: 10.32    us, B avg: 6.53     us, uplift: 58.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=8 n=79 k=5120 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (8, 79, 5120)        dtype: torch.bfloat16, torch avg: 10.28    us, B avg: 9.78     us, uplift: 5.1% [checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=9 n=1 k=8 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (9, 1, 8)            dtype: torch.float16, torch avg: 7.06     us, B avg: 2.39     us, uplift: 195.8%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=9 n=1 k=8 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (9, 1, 8)            dtype: torch.bfloat16, torch avg: 7.18     us, B avg: 3.20     us, uplift: 124.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=9 n=1 k=256 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (9, 1, 256)          dtype: torch.float16, torch avg: 9.40     us, B avg: 2.34     us, uplift: 302.0%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=9 n=1 k=256 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (9, 1, 256)          dtype: torch.bfloat16, torch avg: 9.47     us, B avg: 3.15     us, uplift: 201.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=9 n=80 k=8 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (9, 80, 8)           dtype: torch.float16, torch avg: 9.25     us, B avg: 3.05     us, uplift: 202.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=9 n=80 k=8 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (9, 80, 8)           dtype: torch.bfloat16, torch avg: 9.28     us, B avg: 3.85     us, uplift: 140.7%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=9 n=80 k=256 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (9, 80, 256)         dtype: torch.float16, torch avg: 11.99    us, B avg: 2.88     us, uplift: 316.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=9 n=80 k=256 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (9, 80, 256)         dtype: torch.bfloat16, torch avg: 12.13    us, B avg: 3.72     us, uplift: 225.8%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=16 n=1 k=8 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (16, 1, 8)           dtype: torch.float16, torch avg: 7.25     us, B avg: 3.16     us, uplift: 129.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=16 n=1 k=8 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (16, 1, 8)           dtype: torch.bfloat16, torch avg: 7.26     us, B avg: 4.77     us, uplift: 52.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=16 n=1 k=256 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (16, 1, 256)         dtype: torch.float16, torch avg: 9.61     us, B avg: 3.08     us, uplift: 212.0%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=16 n=1 k=256 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (16, 1, 256)         dtype: torch.bfloat16, torch avg: 9.57     us, B avg: 4.75     us, uplift: 101.8%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=16 n=80 k=8 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (16, 80, 8)          dtype: torch.float16, torch avg: 9.26     us, B avg: 3.77     us, uplift: 145.5%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=16 n=80 k=8 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (16, 80, 8)          dtype: torch.bfloat16, torch avg: 9.30     us, B avg: 5.34     us, uplift: 74.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=16 n=80 k=256 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (16, 80, 256)        dtype: torch.float16, torch avg: 12.93    us, B avg: 3.60     us, uplift: 259.3%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=16 n=80 k=256 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (16, 80, 256)        dtype: torch.bfloat16, torch avg: 12.91    us, B avg: 5.22     us, uplift: 147.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=15 n=80 k=256 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (15, 80, 256)        dtype: torch.float16, torch avg: 12.88    us, B avg: 3.46     us, uplift: 271.7%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=15 n=80 k=256 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (15, 80, 256)        dtype: torch.bfloat16, torch avg: 13.06    us, B avg: 4.97     us, uplift: 162.8%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=16 n=79 k=256 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (16, 79, 256)        dtype: torch.float16, torch avg: 13.07    us, B avg: 3.58     us, uplift: 265.5%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=16 n=79 k=256 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (16, 79, 256)        dtype: torch.bfloat16, torch avg: 12.79    us, B avg: 5.23     us, uplift: 144.6%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=15 n=79 k=256 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (15, 79, 256)        dtype: torch.float16, torch avg: 13.07    us, B avg: 3.52     us, uplift: 271.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=15 n=79 k=256 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (15, 79, 256)        dtype: torch.bfloat16, torch avg: 13.05    us, B avg: 4.96     us, uplift: 163.3%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 80, 9216)        dtype: torch.float16, torch avg: 32.59    us, B avg: 7.15     us, uplift: 355.5%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 80, 9216)        dtype: torch.bfloat16, torch avg: 146.69   us, B avg: 10.17    us, uplift: 1342.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (5, 80, 5120)        dtype: torch.float16, torch avg: 10.23    us, B avg: 5.30     us, uplift: 93.0%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (5, 80, 5120)        dtype: torch.bfloat16, torch avg: 10.20    us, B avg: 7.54     us, uplift: 35.3%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (8, 80, 5120)        dtype: torch.float16, torch avg: 10.28    us, B avg: 6.58     us, uplift: 56.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (8, 80, 5120)        dtype: torch.bfloat16, torch avg: 10.22    us, B avg: 9.88     us, uplift: 3.4% [checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (9, 80, 256)         dtype: torch.float16, torch avg: 12.00    us, B avg: 2.88     us, uplift: 316.5%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (9, 80, 256)         dtype: torch.bfloat16, torch avg: 12.17    us, B avg: 3.71     us, uplift: 227.8%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=1 n=2 k=5240 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (1, 2, 5240)         dtype: torch.float16, torch avg: 12.09    us, B avg: 3.70     us, uplift: 226.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=1 n=2 k=5240 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (1, 2, 5240)         dtype: torch.bfloat16, torch avg: 12.18    us, B avg: 4.22     us, uplift: 188.8%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=1 n=6 k=8968 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (1, 6, 8968)         dtype: torch.float16, torch avg: 12.43    us, B avg: 5.33     us, uplift: 133.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=1 n=6 k=8968 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (1, 6, 8968)         dtype: torch.bfloat16, torch avg: 12.49    us, B avg: 6.18     us, uplift: 101.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 9, 3096)         dtype: torch.float16, torch avg: 5.28     us, B avg: 3.14     us, uplift: 68.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 9, 3096)         dtype: torch.bfloat16, torch avg: 5.88     us, B avg: 3.53     us, uplift: 66.7%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 20, 9120)        dtype: torch.float16, torch avg: 8.18     us, B avg: 5.38     us, uplift: 52.0%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 20, 9120)        dtype: torch.bfloat16, torch avg: 9.92     us, B avg: 6.22     us, uplift: 59.5%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 22, 2048)        dtype: torch.float16, torch avg: 3.49     us, B avg: 2.44     us, uplift: 43.0%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 22, 2048)        dtype: torch.bfloat16, torch avg: 3.95     us, B avg: 2.66     us, uplift: 48.8%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 42, 3400)        dtype: torch.float16, torch avg: 5.12     us, B avg: 3.30     us, uplift: 55.3%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 42, 3400)        dtype: torch.bfloat16, torch avg: 5.80     us, B avg: 3.63     us, uplift: 60.0%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 46, 2368)        dtype: torch.float16, torch avg: 4.42     us, B avg: 3.05     us, uplift: 44.7%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 46, 2368)        dtype: torch.bfloat16, torch avg: 5.02     us, B avg: 3.31     us, uplift: 51.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 47, 1464)        dtype: torch.float16, torch avg: 4.81     us, B avg: 2.45     us, uplift: 96.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 47, 1464)        dtype: torch.bfloat16, torch avg: 4.46     us, B avg: 2.65     us, uplift: 68.6%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 47, 7408)        dtype: torch.float16, torch avg: 8.67     us, B avg: 4.78     us, uplift: 81.5%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 47, 7408)        dtype: torch.bfloat16, torch avg: 9.88     us, B avg: 5.51     us, uplift: 79.3%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 52, 4784)        dtype: torch.float16, torch avg: 5.89     us, B avg: 3.98     us, uplift: 48.0%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 52, 4784)        dtype: torch.bfloat16, torch avg: 6.71     us, B avg: 4.48     us, uplift: 50.0%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 52, 6824)        dtype: torch.float16, torch avg: 7.01     us, B avg: 4.69     us, uplift: 49.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 52, 6824)        dtype: torch.bfloat16, torch avg: 8.37     us, B avg: 5.37     us, uplift: 55.8%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 53, 4048)        dtype: torch.float16, torch avg: 5.89     us, B avg: 3.36     us, uplift: 75.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 53, 4048)        dtype: torch.bfloat16, torch avg: 6.64     us, B avg: 3.75     us, uplift: 77.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 55, 9072)        dtype: torch.float16, torch avg: 9.70     us, B avg: 5.43     us, uplift: 78.5%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 55, 9072)        dtype: torch.bfloat16, torch avg: 11.45    us, B avg: 6.31     us, uplift: 81.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 60, 5256)        dtype: torch.float16, torch avg: 6.30     us, B avg: 4.03     us, uplift: 56.5%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 60, 5256)        dtype: torch.bfloat16, torch avg: 7.31     us, B avg: 4.55     us, uplift: 60.7%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 67, 4168)        dtype: torch.float16, torch avg: 6.36     us, B avg: 3.89     us, uplift: 63.6%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 67, 4168)        dtype: torch.bfloat16, torch avg: 7.24     us, B avg: 4.30     us, uplift: 68.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 68, 2832)        dtype: torch.float16, torch avg: 4.50     us, B avg: 3.23     us, uplift: 39.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 68, 2832)        dtype: torch.bfloat16, torch avg: 5.21     us, B avg: 3.57     us, uplift: 46.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 74, 456)         dtype: torch.float16, torch avg: 2.83     us, B avg: 2.28     us, uplift: 24.5%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 74, 456)         dtype: torch.bfloat16, torch avg: 3.09     us, B avg: 2.37     us, uplift: 30.3%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 74, 1616)        dtype: torch.float16, torch avg: 3.85     us, B avg: 2.51     us, uplift: 53.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 74, 1616)        dtype: torch.bfloat16, torch avg: 4.30     us, B avg: 2.73     us, uplift: 57.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 78, 6104)        dtype: torch.float16, torch avg: 6.46     us, B avg: 4.14     us, uplift: 56.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 78, 6104)        dtype: torch.bfloat16, torch avg: 7.66     us, B avg: 4.69     us, uplift: 63.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 80, 6840)        dtype: torch.float16, torch avg: 7.07     us, B avg: 4.71     us, uplift: 50.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 80, 6840)        dtype: torch.bfloat16, torch avg: 8.50     us, B avg: 5.39     us, uplift: 57.7%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 94, 2272)        dtype: torch.float16, torch avg: 4.30     us, B avg: 4.54     us, uplift: -5.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 94, 2272)        dtype: torch.bfloat16, torch avg: 4.85     us, B avg: 5.07     us, uplift: -4.3%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 94, 7880)        dtype: torch.float16, torch avg: 7.77     us, B avg: 8.08     us, uplift: -3.8%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 94, 7880)        dtype: torch.bfloat16, torch avg: 9.47     us, B avg: 9.56     us, uplift: -0.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 98, 8104)        dtype: torch.float16, torch avg: 7.75     us, B avg: 8.19     us, uplift: -5.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 98, 8104)        dtype: torch.bfloat16, torch avg: 9.52     us, B avg: 9.63     us, uplift: -1.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 99, 2032)        dtype: torch.float16, torch avg: 4.20     us, B avg: 3.52     us, uplift: 19.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 99, 2032)        dtype: torch.bfloat16, torch avg: 4.70     us, B avg: 3.94     us, uplift: 19.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 102, 2656)       dtype: torch.float16, torch avg: 4.50     us, B avg: 4.77     us, uplift: -5.6%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 102, 2656)       dtype: torch.bfloat16, torch avg: 5.09     us, B avg: 5.37     us, uplift: -5.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 102, 3576)       dtype: torch.float16, torch avg: 5.08     us, B avg: 5.02     us, uplift: 1.0% [checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 102, 3576)       dtype: torch.bfloat16, torch avg: 5.76     us, B avg: 5.71     us, uplift: 0.9% [checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 102, 5872)       dtype: torch.float16, torch avg: 6.51     us, B avg: 6.66     us, uplift: -2.3%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 102, 5872)       dtype: torch.bfloat16, torch avg: 7.80     us, B avg: 7.75     us, uplift: 0.7% [checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 105, 8440)       dtype: torch.float16, torch avg: 9.67     us, B avg: 9.04     us, uplift: 7.0% [checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 105, 8440)       dtype: torch.bfloat16, torch avg: 11.53    us, B avg: 10.37    us, uplift: 11.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 109, 976)        dtype: torch.float16, torch avg: 3.31     us, B avg: 3.23     us, uplift: 2.3% [checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 109, 976)        dtype: torch.bfloat16, torch avg: 3.72     us, B avg: 3.50     us, uplift: 6.1% [checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 110, 4656)       dtype: torch.float16, torch avg: 6.08     us, B avg: 6.31     us, uplift: -3.7%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 110, 4656)       dtype: torch.bfloat16, torch avg: 6.88     us, B avg: 7.31     us, uplift: -6.0%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 118, 5344)       dtype: torch.float16, torch avg: 6.32     us, B avg: 6.59     us, uplift: -4.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 118, 5344)       dtype: torch.bfloat16, torch avg: 7.43     us, B avg: 7.56     us, uplift: -1.8%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 128, 24)         dtype: torch.float16, torch avg: 2.79     us, B avg: 3.04     us, uplift: -7.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 128, 24)         dtype: torch.bfloat16, torch avg: 3.04     us, B avg: 3.15     us, uplift: -3.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 129, 1496)       dtype: torch.float16, torch avg: 3.83     us, B avg: 3.40     us, uplift: 12.5%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 129, 1496)       dtype: torch.bfloat16, torch avg: 4.34     us, B avg: 3.73     us, uplift: 16.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 130, 4712)       dtype: torch.float16, torch avg: 6.03     us, B avg: 6.39     us, uplift: -5.7%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 130, 4712)       dtype: torch.bfloat16, torch avg: 6.89     us, B avg: 7.33     us, uplift: -5.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 133, 7824)       dtype: torch.float16, torch avg: 8.92     us, B avg: 8.14     us, uplift: 9.6% [checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 133, 7824)       dtype: torch.bfloat16, torch avg: 10.67    us, B avg: 9.57     us, uplift: 11.5%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 136, 1528)       dtype: torch.float16, torch avg: 3.64     us, B avg: 3.55     us, uplift: 2.6% [checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 136, 1528)       dtype: torch.bfloat16, torch avg: 4.11     us, B avg: 3.89     us, uplift: 5.6% [checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 151, 7976)       dtype: torch.float16, torch avg: 9.16     us, B avg: 8.32     us, uplift: 10.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 151, 7976)       dtype: torch.bfloat16, torch avg: 10.73    us, B avg: 9.76     us, uplift: 9.9% [checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 153, 696)        dtype: torch.float16, torch avg: 3.29     us, B avg: 3.22     us, uplift: 2.0% [checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 153, 696)        dtype: torch.bfloat16, torch avg: 3.70     us, B avg: 3.50     us, uplift: 5.8% [checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 158, 792)        dtype: torch.float16, torch avg: 3.11     us, B avg: 3.37     us, uplift: -7.6%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 158, 792)        dtype: torch.bfloat16, torch avg: 3.49     us, B avg: 3.65     us, uplift: -4.3%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 158, 1344)       dtype: torch.float16, torch avg: 3.69     us, B avg: 3.41     us, uplift: 8.0% [checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (1, 158, 1344)       dtype: torch.bfloat16, torch avg: 4.11     us, B avg: 3.77     us, uplift: 8.9% [checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=2 n=1 k=5736 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (2, 1, 5736)         dtype: torch.float16, torch avg: 12.13    us, B avg: 3.78     us, uplift: 220.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=2 n=1 k=5736 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (2, 1, 5736)         dtype: torch.bfloat16, torch avg: 12.06    us, B avg: 5.16     us, uplift: 133.5%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=2 n=4 k=1168 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (2, 4, 1168)         dtype: torch.float16, torch avg: 11.01    us, B avg: 2.42     us, uplift: 355.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=2 n=4 k=1168 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (2, 4, 1168)         dtype: torch.bfloat16, torch avg: 10.99    us, B avg: 2.84     us, uplift: 287.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=2 n=6 k=176 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (2, 6, 176)          dtype: torch.float16, torch avg: 9.50     us, B avg: 2.25     us, uplift: 322.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=2 n=6 k=176 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (2, 6, 176)          dtype: torch.bfloat16, torch avg: 9.57     us, B avg: 2.51     us, uplift: 281.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=2 n=8 k=8280 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (2, 8, 8280)         dtype: torch.float16, torch avg: 12.90    us, B avg: 5.53     us, uplift: 133.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=2 n=8 k=8280 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (2, 8, 8280)         dtype: torch.bfloat16, torch avg: 12.80    us, B avg: 7.39     us, uplift: 73.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 11, 3512)        dtype: torch.float16, torch avg: 6.28     us, B avg: 3.43     us, uplift: 83.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 11, 3512)        dtype: torch.bfloat16, torch avg: 7.79     us, B avg: 4.28     us, uplift: 81.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 16, 2224)        dtype: torch.float16, torch avg: 4.43     us, B avg: 3.08     us, uplift: 43.6%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 16, 2224)        dtype: torch.bfloat16, torch avg: 5.55     us, B avg: 3.74     us, uplift: 48.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 21, 1784)        dtype: torch.float16, torch avg: 4.62     us, B avg: 2.64     us, uplift: 75.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 21, 1784)        dtype: torch.bfloat16, torch avg: 5.66     us, B avg: 3.18     us, uplift: 78.0%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 22, 2632)        dtype: torch.float16, torch avg: 4.68     us, B avg: 3.28     us, uplift: 42.7%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 22, 2632)        dtype: torch.bfloat16, torch avg: 5.95     us, B avg: 4.05     us, uplift: 46.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 25, 8792)        dtype: torch.float16, torch avg: 12.68    us, B avg: 5.75     us, uplift: 120.7%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 25, 8792)        dtype: torch.bfloat16, torch avg: 16.23    us, B avg: 7.73     us, uplift: 110.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 26, 976)         dtype: torch.float16, torch avg: 3.18     us, B avg: 2.43     us, uplift: 30.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 26, 976)         dtype: torch.bfloat16, torch avg: 3.89     us, B avg: 2.81     us, uplift: 38.5%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 30, 7208)        dtype: torch.float16, torch avg: 8.47     us, B avg: 5.10     us, uplift: 66.0%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 30, 7208)        dtype: torch.bfloat16, torch avg: 12.21    us, B avg: 6.75     us, uplift: 80.7%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 31, 4792)        dtype: torch.float16, torch avg: 8.01     us, B avg: 4.15     us, uplift: 92.8%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 31, 4792)        dtype: torch.bfloat16, torch avg: 10.26    us, B avg: 5.29     us, uplift: 93.8%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 37, 104)         dtype: torch.float16, torch avg: 3.30     us, B avg: 2.37     us, uplift: 39.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 37, 104)         dtype: torch.bfloat16, torch avg: 3.72     us, B avg: 2.62     us, uplift: 42.0%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 38, 1824)        dtype: torch.float16, torch avg: 3.99     us, B avg: 2.65     us, uplift: 50.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 38, 1824)        dtype: torch.bfloat16, torch avg: 5.06     us, B avg: 3.19     us, uplift: 58.5%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 43, 1424)        dtype: torch.float16, torch avg: 4.66     us, B avg: 2.59     us, uplift: 80.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 43, 1424)        dtype: torch.bfloat16, torch avg: 5.48     us, B avg: 3.04     us, uplift: 80.3%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 47, 4600)        dtype: torch.float16, torch avg: 7.82     us, B avg: 4.09     us, uplift: 91.3%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 47, 4600)        dtype: torch.bfloat16, torch avg: 9.87     us, B avg: 5.14     us, uplift: 92.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 50, 1840)        dtype: torch.float16, torch avg: 4.15     us, B avg: 2.66     us, uplift: 56.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 50, 1840)        dtype: torch.bfloat16, torch avg: 5.11     us, B avg: 3.23     us, uplift: 58.5%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 52, 2320)        dtype: torch.float16, torch avg: 4.82     us, B avg: 3.79     us, uplift: 27.3%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 52, 2320)        dtype: torch.bfloat16, torch avg: 5.86     us, B avg: 3.85     us, uplift: 52.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 52, 4232)        dtype: torch.float16, torch avg: 6.41     us, B avg: 4.06     us, uplift: 57.8%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 52, 4232)        dtype: torch.bfloat16, torch avg: 8.41     us, B avg: 5.13     us, uplift: 64.0%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 55, 352)         dtype: torch.float16, torch avg: 3.34     us, B avg: 2.37     us, uplift: 40.8%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 55, 352)         dtype: torch.bfloat16, torch avg: 3.83     us, B avg: 2.63     us, uplift: 45.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 57, 2368)        dtype: torch.float16, torch avg: 5.70     us, B avg: 3.21     us, uplift: 77.7%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 57, 2368)        dtype: torch.bfloat16, torch avg: 7.06     us, B avg: 3.85     us, uplift: 83.3%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 64, 7648)        dtype: torch.float16, torch avg: 8.66     us, B avg: 5.13     us, uplift: 68.7%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 64, 7648)        dtype: torch.bfloat16, torch avg: 11.91    us, B avg: 6.77     us, uplift: 75.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 66, 1280)        dtype: torch.float16, torch avg: 3.89     us, B avg: 2.48     us, uplift: 56.8%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 66, 1280)        dtype: torch.bfloat16, torch avg: 4.75     us, B avg: 2.93     us, uplift: 61.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 73, 2536)        dtype: torch.float16, torch avg: 5.66     us, B avg: 3.19     us, uplift: 77.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 73, 2536)        dtype: torch.bfloat16, torch avg: 7.01     us, B avg: 3.89     us, uplift: 80.3%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 76, 4904)        dtype: torch.float16, torch avg: 6.71     us, B avg: 4.22     us, uplift: 58.8%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 76, 4904)        dtype: torch.bfloat16, torch avg: 8.91     us, B avg: 5.37     us, uplift: 65.8%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 78, 464)         dtype: torch.float16, torch avg: 3.02     us, B avg: 2.34     us, uplift: 28.8%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 78, 464)         dtype: torch.bfloat16, torch avg: 3.48     us, B avg: 2.61     us, uplift: 33.3%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 78, 6400)        dtype: torch.float16, torch avg: 7.90     us, B avg: 4.78     us, uplift: 65.3%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (2, 78, 6400)        dtype: torch.bfloat16, torch avg: 10.74    us, B avg: 6.27     us, uplift: 71.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=3 n=7 k=8984 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (3, 7, 8984)         dtype: torch.float16, torch avg: 12.86    us, B avg: 6.46     us, uplift: 99.0%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=3 n=7 k=8984 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (3, 7, 8984)         dtype: torch.bfloat16, torch avg: 12.73    us, B avg: 9.06     us, uplift: 40.5%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (3, 10, 624)         dtype: torch.float16, torch avg: 5.99     us, B avg: 2.50     us, uplift: 139.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (3, 10, 624)         dtype: torch.bfloat16, torch avg: 16.72    us, B avg: 2.98     us, uplift: 460.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (3, 13, 6840)        dtype: torch.float16, torch avg: 19.56    us, B avg: 5.45     us, uplift: 258.8%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (3, 13, 6840)        dtype: torch.bfloat16, torch avg: 53.85    us, B avg: 7.52     us, uplift: 615.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (3, 14, 6696)        dtype: torch.float16, torch avg: 19.27    us, B avg: 5.52     us, uplift: 249.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (3, 14, 6696)        dtype: torch.bfloat16, torch avg: 54.32    us, B avg: 7.56     us, uplift: 618.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (3, 22, 2624)        dtype: torch.float16, torch avg: 10.71    us, B avg: 3.48     us, uplift: 207.6%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (3, 22, 2624)        dtype: torch.bfloat16, torch avg: 33.14    us, B avg: 4.50     us, uplift: 635.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (3, 26, 1032)        dtype: torch.float16, torch avg: 7.20     us, B avg: 2.69     us, uplift: 167.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (3, 26, 1032)        dtype: torch.bfloat16, torch avg: 20.76    us, B avg: 3.27     us, uplift: 535.0%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (3, 33, 1504)        dtype: torch.float16, torch avg: 7.92     us, B avg: 2.71     us, uplift: 192.0%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (3, 33, 1504)        dtype: torch.bfloat16, torch avg: 22.31    us, B avg: 3.29     us, uplift: 577.5%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (3, 40, 7032)        dtype: torch.float16, torch avg: 13.01    us, B avg: 5.52     us, uplift: 135.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (3, 40, 7032)        dtype: torch.bfloat16, torch avg: 22.19    us, B avg: 7.55     us, uplift: 194.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (3, 50, 8440)        dtype: torch.float16, torch avg: 26.83    us, B avg: 6.32     us, uplift: 324.6%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (3, 50, 8440)        dtype: torch.bfloat16, torch avg: 84.79    us, B avg: 8.77     us, uplift: 867.0%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (3, 71, 8056)        dtype: torch.float16, torch avg: 23.74    us, B avg: 5.90     us, uplift: 302.6%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (3, 71, 8056)        dtype: torch.bfloat16, torch avg: 79.51    us, B avg: 8.23     us, uplift: 866.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (3, 80, 8384)        dtype: torch.float16, torch avg: 14.75    us, B avg: 6.31     us, uplift: 133.6%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (3, 80, 8384)        dtype: torch.bfloat16, torch avg: 25.80    us, B avg: 8.75     us, uplift: 194.7%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=4 n=2 k=5872 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (4, 2, 5872)         dtype: torch.float16, torch avg: 12.61    us, B avg: 5.05     us, uplift: 149.7%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=4 n=2 k=5872 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (4, 2, 5872)         dtype: torch.bfloat16, torch avg: 12.66    us, B avg: 7.24     us, uplift: 74.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=4 n=3 k=1768 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (4, 3, 1768)         dtype: torch.float16, torch avg: 11.67    us, B avg: 2.91     us, uplift: 301.3%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=4 n=3 k=1768 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (4, 3, 1768)         dtype: torch.bfloat16, torch avg: 11.87    us, B avg: 3.74     us, uplift: 217.7%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=4 n=3 k=4072 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (4, 3, 4072)         dtype: torch.float16, torch avg: 14.16    us, B avg: 4.03     us, uplift: 251.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=4 n=3 k=4072 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (4, 3, 4072)         dtype: torch.bfloat16, torch avg: 14.40    us, B avg: 5.56     us, uplift: 158.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=4 n=6 k=6384 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (4, 6, 6384)         dtype: torch.float16, torch avg: 12.49    us, B avg: 5.79     us, uplift: 115.7%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=4 n=6 k=6384 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (4, 6, 6384)         dtype: torch.bfloat16, torch avg: 12.49    us, B avg: 8.11     us, uplift: 54.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=4 n=8 k=1896 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (4, 8, 1896)         dtype: torch.float16, torch avg: 12.31    us, B avg: 3.00     us, uplift: 309.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=4 n=8 k=1896 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (4, 8, 1896)         dtype: torch.bfloat16, torch avg: 12.51    us, B avg: 3.84     us, uplift: 225.5%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 9, 5496)         dtype: torch.float16, torch avg: 18.92    us, B avg: 5.16     us, uplift: 266.8%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 9, 5496)         dtype: torch.bfloat16, torch avg: 78.49    us, B avg: 7.15     us, uplift: 998.5%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 13, 9048)        dtype: torch.float16, torch avg: 29.92    us, B avg: 7.16     us, uplift: 318.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 13, 9048)        dtype: torch.bfloat16, torch avg: 121.84   us, B avg: 10.14    us, uplift: 1101.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 16, 7280)        dtype: torch.float16, torch avg: 14.52    us, B avg: 6.31     us, uplift: 130.0%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 16, 7280)        dtype: torch.bfloat16, torch avg: 22.83    us, B avg: 8.91     us, uplift: 156.3%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 17, 8192)        dtype: torch.float16, torch avg: 25.11    us, B avg: 6.26     us, uplift: 301.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 17, 8192)        dtype: torch.bfloat16, torch avg: 110.18   us, B avg: 8.99     us, uplift: 1125.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 31, 4720)        dtype: torch.float16, torch avg: 19.09    us, B avg: 4.97     us, uplift: 283.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 31, 4720)        dtype: torch.bfloat16, torch avg: 80.18    us, B avg: 6.84     us, uplift: 1071.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 46, 608)         dtype: torch.float16, torch avg: 7.73     us, B avg: 2.76     us, uplift: 180.5%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 46, 608)         dtype: torch.bfloat16, torch avg: 30.32    us, B avg: 3.28     us, uplift: 823.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 47, 3328)        dtype: torch.float16, torch avg: 15.13    us, B avg: 3.95     us, uplift: 282.7%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 47, 3328)        dtype: torch.bfloat16, torch avg: 67.45    us, B avg: 5.30     us, uplift: 1172.6%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 47, 6064)        dtype: torch.float16, torch avg: 22.63    us, B avg: 5.43     us, uplift: 316.8%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 47, 6064)        dtype: torch.bfloat16, torch avg: 103.90   us, B avg: 7.54     us, uplift: 1278.6%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 49, 208)         dtype: torch.float16, torch avg: 6.30     us, B avg: 2.61     us, uplift: 141.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 49, 208)         dtype: torch.bfloat16, torch avg: 25.16    us, B avg: 2.99     us, uplift: 742.6%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 54, 3704)        dtype: torch.float16, torch avg: 17.15    us, B avg: 4.29     us, uplift: 300.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 54, 3704)        dtype: torch.bfloat16, torch avg: 74.36    us, B avg: 5.83     us, uplift: 1174.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 59, 1096)        dtype: torch.float16, torch avg: 9.63     us, B avg: 2.95     us, uplift: 226.7%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 59, 1096)        dtype: torch.bfloat16, torch avg: 37.77    us, B avg: 3.66     us, uplift: 930.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 59, 1712)        dtype: torch.float16, torch avg: 11.42    us, B avg: 3.12     us, uplift: 265.6%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 59, 1712)        dtype: torch.bfloat16, torch avg: 45.30    us, B avg: 3.95     us, uplift: 1048.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 62, 4104)        dtype: torch.float16, torch avg: 18.28    us, B avg: 4.60     us, uplift: 297.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 62, 4104)        dtype: torch.bfloat16, torch avg: 80.84    us, B avg: 6.26     us, uplift: 1191.7%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 63, 1384)        dtype: torch.float16, torch avg: 9.41     us, B avg: 2.88     us, uplift: 226.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 63, 1384)        dtype: torch.bfloat16, torch avg: 38.64    us, B avg: 3.61     us, uplift: 970.0%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 64, 1376)        dtype: torch.float16, torch avg: 5.88     us, B avg: 2.95     us, uplift: 99.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 64, 1376)        dtype: torch.bfloat16, torch avg: 8.99     us, B avg: 3.67     us, uplift: 145.0%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=5 n=5 k=4456 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (5, 5, 4456)         dtype: torch.float16, torch avg: 15.52    us, B avg: 4.99     us, uplift: 210.8%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=5 n=5 k=4456 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (5, 5, 4456)         dtype: torch.bfloat16, torch avg: 15.30    us, B avg: 7.05     us, uplift: 117.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=5 n=8 k=3880 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (5, 8, 3880)         dtype: torch.float16, torch avg: 14.77    us, B avg: 4.49     us, uplift: 228.8%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=5 n=8 k=3880 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (5, 8, 3880)         dtype: torch.bfloat16, torch avg: 14.88    us, B avg: 6.27     us, uplift: 137.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=5 n=18 k=1832 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (5, 18, 1832)        dtype: torch.float16, torch avg: 13.51    us, B avg: 3.28     us, uplift: 311.8%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=5 n=18 k=1832 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (5, 18, 1832)        dtype: torch.bfloat16, torch avg: 13.86    us, B avg: 4.30     us, uplift: 222.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=5 n=21 k=4024 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (5, 21, 4024)        dtype: torch.float16, torch avg: 16.01    us, B avg: 4.46     us, uplift: 259.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=5 n=21 k=4024 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (5, 21, 4024)        dtype: torch.bfloat16, torch avg: 15.93    us, B avg: 6.27     us, uplift: 154.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=5 n=24 k=904 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (5, 24, 904)         dtype: torch.float16, torch avg: 12.40    us, B avg: 2.76     us, uplift: 349.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=5 n=24 k=904 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (5, 24, 904)         dtype: torch.bfloat16, torch avg: 12.72    us, B avg: 3.42     us, uplift: 271.7%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=5 n=56 k=1144 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (5, 56, 1144)        dtype: torch.float16, torch avg: 13.35    us, B avg: 3.13     us, uplift: 326.5%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=5 n=56 k=1144 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (5, 56, 1144)        dtype: torch.bfloat16, torch avg: 13.55    us, B avg: 3.99     us, uplift: 239.7%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=5 n=57 k=1992 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (5, 57, 1992)        dtype: torch.float16, torch avg: 14.23    us, B avg: 3.27     us, uplift: 334.5%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=5 n=57 k=1992 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (5, 57, 1992)        dtype: torch.bfloat16, torch avg: 14.18    us, B avg: 4.31     us, uplift: 228.8%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=5 n=59 k=2048 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (5, 59, 2048)        dtype: torch.float16, torch avg: 6.62     us, B avg: 3.17     us, uplift: 109.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=5 n=59 k=2048 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (5, 59, 2048)        dtype: torch.bfloat16, torch avg: 6.61     us, B avg: 4.21     us, uplift: 57.0%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=5 n=60 k=3216 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (5, 60, 3216)        dtype: torch.float16, torch avg: 15.57    us, B avg: 4.30     us, uplift: 262.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=5 n=60 k=3216 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (5, 60, 3216)        dtype: torch.bfloat16, torch avg: 15.53    us, B avg: 5.98     us, uplift: 159.6%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=5 n=64 k=4608 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (5, 64, 4608)        dtype: torch.float16, torch avg: 9.46     us, B avg: 5.01     us, uplift: 89.0%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=5 n=64 k=4608 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (5, 64, 4608)        dtype: torch.bfloat16, torch avg: 9.49     us, B avg: 7.04     us, uplift: 34.8%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=5 n=67 k=200 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (5, 67, 200)         dtype: torch.float16, torch avg: 11.42    us, B avg: 2.59     us, uplift: 341.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=5 n=67 k=200 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (5, 67, 200)         dtype: torch.bfloat16, torch avg: 11.61    us, B avg: 3.07     us, uplift: 278.0%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=5 n=68 k=176 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (5, 68, 176)         dtype: torch.float16, torch avg: 11.61    us, B avg: 2.59     us, uplift: 348.8%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=5 n=68 k=176 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (5, 68, 176)         dtype: torch.bfloat16, torch avg: 11.84    us, B avg: 3.05     us, uplift: 288.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=5 n=73 k=3376 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (5, 73, 3376)        dtype: torch.float16, torch avg: 15.75    us, B avg: 4.40     us, uplift: 258.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=5 n=73 k=3376 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (5, 73, 3376)        dtype: torch.bfloat16, torch avg: 15.68    us, B avg: 6.05     us, uplift: 159.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=6 n=9 k=4392 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (6, 9, 4392)         dtype: torch.float16, torch avg: 15.90    us, B avg: 5.44     us, uplift: 192.6%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=6 n=9 k=4392 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (6, 9, 4392)         dtype: torch.bfloat16, torch avg: 15.79    us, B avg: 7.73     us, uplift: 104.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=6 n=13 k=3304 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (6, 13, 3304)        dtype: torch.float16, torch avg: 15.01    us, B avg: 4.64     us, uplift: 223.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=6 n=13 k=3304 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (6, 13, 3304)        dtype: torch.bfloat16, torch avg: 15.18    us, B avg: 6.59     us, uplift: 130.3%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=6 n=44 k=2744 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (6, 44, 2744)        dtype: torch.float16, torch avg: 14.93    us, B avg: 4.37     us, uplift: 241.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=6 n=44 k=2744 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (6, 44, 2744)        dtype: torch.bfloat16, torch avg: 14.95    us, B avg: 6.11     us, uplift: 144.7%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=6 n=49 k=4440 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (6, 49, 4440)        dtype: torch.float16, torch avg: 16.92    us, B avg: 5.53     us, uplift: 206.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=6 n=49 k=4440 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (6, 49, 4440)        dtype: torch.bfloat16, torch avg: 16.85    us, B avg: 7.86     us, uplift: 114.3%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=6 n=54 k=1328 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (6, 54, 1328)        dtype: torch.float16, torch avg: 13.39    us, B avg: 3.22     us, uplift: 315.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=6 n=54 k=1328 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (6, 54, 1328)        dtype: torch.bfloat16, torch avg: 13.68    us, B avg: 4.29     us, uplift: 219.0%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=6 n=55 k=1248 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (6, 55, 1248)        dtype: torch.float16, torch avg: 13.59    us, B avg: 3.20     us, uplift: 325.3%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=6 n=55 k=1248 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (6, 55, 1248)        dtype: torch.bfloat16, torch avg: 13.58    us, B avg: 4.23     us, uplift: 220.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=6 n=74 k=2464 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (6, 74, 2464)        dtype: torch.float16, torch avg: 14.81    us, B avg: 4.15     us, uplift: 257.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=6 n=74 k=2464 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (6, 74, 2464)        dtype: torch.bfloat16, torch avg: 14.76    us, B avg: 5.59     us, uplift: 163.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=7 n=5 k=576 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (7, 5, 576)          dtype: torch.float16, torch avg: 10.44    us, B avg: 2.98     us, uplift: 250.7%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=7 n=5 k=576 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (7, 5, 576)          dtype: torch.bfloat16, torch avg: 10.65    us, B avg: 3.88     us, uplift: 174.5%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=7 n=5 k=808 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (7, 5, 808)          dtype: torch.float16, torch avg: 11.39    us, B avg: 3.05     us, uplift: 274.0%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=7 n=5 k=808 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (7, 5, 808)          dtype: torch.bfloat16, torch avg: 11.48    us, B avg: 3.97     us, uplift: 189.0%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=7 n=16 k=3296 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (7, 16, 3296)        dtype: torch.float16, torch avg: 15.48    us, B avg: 4.89     us, uplift: 216.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=7 n=16 k=3296 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (7, 16, 3296)        dtype: torch.bfloat16, torch avg: 15.42    us, B avg: 7.14     us, uplift: 116.0%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=7 n=52 k=1400 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (7, 52, 1400)        dtype: torch.float16, torch avg: 13.54    us, B avg: 3.44     us, uplift: 293.3%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=7 n=52 k=1400 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (7, 52, 1400)        dtype: torch.bfloat16, torch avg: 13.75    us, B avg: 4.65     us, uplift: 195.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=7 n=52 k=3032 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (7, 52, 3032)        dtype: torch.float16, torch avg: 15.45    us, B avg: 4.65     us, uplift: 232.3%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=7 n=52 k=3032 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (7, 52, 3032)        dtype: torch.bfloat16, torch avg: 15.32    us, B avg: 6.70     us, uplift: 128.8%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=7 n=64 k=3848 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (7, 64, 3848)        dtype: torch.float16, torch avg: 16.30    us, B avg: 5.27     us, uplift: 209.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=7 n=64 k=3848 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (7, 64, 3848)        dtype: torch.bfloat16, torch avg: 16.31    us, B avg: 7.73     us, uplift: 110.8%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=7 n=68 k=3736 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (7, 68, 3736)        dtype: torch.float16, torch avg: 16.19    us, B avg: 5.23     us, uplift: 209.5%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=7 n=68 k=3736 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (7, 68, 3736)        dtype: torch.bfloat16, torch avg: 16.20    us, B avg: 7.73     us, uplift: 109.6%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=7 n=69 k=3712 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (7, 69, 3712)        dtype: torch.float16, torch avg: 16.18    us, B avg: 5.16     us, uplift: 213.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=7 n=69 k=3712 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (7, 69, 3712)        dtype: torch.bfloat16, torch avg: 16.19    us, B avg: 7.66     us, uplift: 111.3%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=7 n=70 k=2136 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (7, 70, 2136)        dtype: torch.float16, torch avg: 14.09    us, B avg: 4.30     us, uplift: 228.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=7 n=70 k=2136 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (7, 70, 2136)        dtype: torch.bfloat16, torch avg: 14.18    us, B avg: 6.11     us, uplift: 131.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=8 n=18 k=3304 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (8, 18, 3304)        dtype: torch.float16, torch avg: 15.65    us, B avg: 5.23     us, uplift: 199.3%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=8 n=18 k=3304 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (8, 18, 3304)        dtype: torch.bfloat16, torch avg: 15.53    us, B avg: 7.70     us, uplift: 101.7%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=8 n=22 k=3768 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (8, 22, 3768)        dtype: torch.float16, torch avg: 15.91    us, B avg: 5.60     us, uplift: 184.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=8 n=22 k=3768 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (8, 22, 3768)        dtype: torch.bfloat16, torch avg: 15.91    us, B avg: 8.25     us, uplift: 92.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=8 n=23 k=4096 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (8, 23, 4096)        dtype: torch.float16, torch avg: 8.80     us, B avg: 5.46     us, uplift: 61.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=8 n=23 k=4096 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (8, 23, 4096)        dtype: torch.bfloat16, torch avg: 8.81     us, B avg: 8.19     us, uplift: 7.6% [checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=8 n=28 k=4120 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (8, 28, 4120)        dtype: torch.float16, torch avg: 14.63    us, B avg: 6.09     us, uplift: 140.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=8 n=28 k=4120 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (8, 28, 4120)        dtype: torch.bfloat16, torch avg: 14.66    us, B avg: 9.08     us, uplift: 61.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=8 n=34 k=1432 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (8, 34, 1432)        dtype: torch.float16, torch avg: 13.50    us, B avg: 3.56     us, uplift: 278.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=8 n=34 k=1432 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (8, 34, 1432)        dtype: torch.bfloat16, torch avg: 13.77    us, B avg: 4.89     us, uplift: 181.6%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=8 n=37 k=4888 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (8, 37, 4888)        dtype: torch.float16, torch avg: 17.44    us, B avg: 6.65     us, uplift: 162.5%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=8 n=37 k=4888 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (8, 37, 4888)        dtype: torch.bfloat16, torch avg: 17.38    us, B avg: 9.88     us, uplift: 75.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=8 n=51 k=4544 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (8, 51, 4544)        dtype: torch.float16, torch avg: 17.02    us, B avg: 6.13     us, uplift: 177.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=8 n=51 k=4544 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (8, 51, 4544)        dtype: torch.bfloat16, torch avg: 17.08    us, B avg: 9.15     us, uplift: 86.5%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=8 n=56 k=3200 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (8, 56, 3200)        dtype: torch.float16, torch avg: 15.78    us, B avg: 5.19     us, uplift: 204.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=8 n=56 k=3200 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (8, 56, 3200)        dtype: torch.bfloat16, torch avg: 15.76    us, B avg: 7.69     us, uplift: 104.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=8 n=59 k=3008 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (8, 59, 3008)        dtype: torch.float16, torch avg: 15.43    us, B avg: 4.89     us, uplift: 215.5%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=8 n=59 k=3008 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (8, 59, 3008)        dtype: torch.bfloat16, torch avg: 15.41    us, B avg: 7.08     us, uplift: 117.7%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=10 n=52 k=72 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (10, 52, 72)         dtype: torch.float16, torch avg: 10.94    us, B avg: 3.01     us, uplift: 264.0%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=10 n=52 k=72 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (10, 52, 72)         dtype: torch.bfloat16, torch avg: 11.17    us, B avg: 4.01     us, uplift: 178.8%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=12 n=28 k=80 dtype=torch.float16 bias=False, scaleAB=False
[aiter] [perf] dim: (12, 28, 80)         dtype: torch.float16, torch avg: 11.16    us, B avg: 3.20     us, uplift: 249.1%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] using soltype=0, solidx=0 for m=12 n=28 k=80 dtype=torch.bfloat16 bias=False, scaleAB=False
[aiter] [perf] dim: (12, 28, 80)         dtype: torch.bfloat16, torch avg: 10.98    us, B avg: 4.29     us, uplift: 156.0%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
