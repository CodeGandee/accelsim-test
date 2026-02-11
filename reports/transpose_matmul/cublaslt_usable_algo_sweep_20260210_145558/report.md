
cuBLASLt usable-algo sweep (int8 square, row-major)
===================================================

# Run Metadata


`2026-02-11T02:45:12.968792+00:00`

Output: `reports/transpose_matmul/cublaslt_usable_algo_sweep_20260210_145558`

Sweep bin: `/data/ssd1/huangzhe/code/accelsim-test/cpp/build/Release/cublaslt_usable_algo_sweep` (pixi env: `cuda13`)

Workspace policy: `max_workspace_bytes=67108864`

Timing policy: `warmup=10`, `iters=50`
# Results (AB vs ABT_view)

## Executive Summary


- N=1000: best AB algo_id=64 (32.85us), best ABT_view algo_id=23 (14.40us) ⇒ ABT_view is ~2.28× faster (best-of-case).
  - algo_id=23: AB=NA (rank NA), ABT_view=14.40us (rank 1/2)
- N=1024: best AB algo_id=70 (7.44us), best ABT_view algo_id=70 (7.66us) ⇒ AB is ~1.03× faster (best-of-case).
  - algo_id=23: AB=NA (rank NA), ABT_view=14.40us (rank 4/6)
- N=2048: best AB algo_id=71 (12.37us), best ABT_view algo_id=71 (12.36us) ⇒ ≈ equal (best-of-case).
  - algo_id=23: AB=NA (rank NA), ABT_view=45.12us (rank 4/6)
## N=1000


| algo_id | AB time_us | ABT_view time_us | ABT/AB |
|------:|----------:|---------------:|------:|
| 23 | NA | 14.40 | NA |
| 64 | 32.85 | 23.91 | 0.728 |

Full merged table: `merged_table.csv` (all algo_id rows).
## N=1024


| algo_id | AB time_us | ABT_view time_us | ABT/AB |
|------:|----------:|---------------:|------:|
| 70 | 7.44 | 7.66 | 1.030 |
| 71 | 7.89 | 7.91 | 1.003 |
| 21 | NA | 8.26 | NA |
| 23 | NA | 14.40 | NA |
| 20 | 28.74 | 26.68 | 0.929 |
| 64 | 38.97 | 32.84 | 0.843 |

Full merged table: `merged_table.csv` (all algo_id rows).
## N=2048


| algo_id | AB time_us | ABT_view time_us | ABT/AB |
|------:|----------:|---------------:|------:|
| 71 | 12.37 | 12.36 | 0.999 |
| 70 | 14.42 | 14.42 | 1.000 |
| 21 | NA | 26.68 | NA |
| 23 | NA | 45.12 | NA |
| 64 | 136.89 | 100.39 | 0.733 |
| 20 | 162.04 | 155.45 | 0.959 |

Full merged table: `merged_table.csv` (all algo_id rows).