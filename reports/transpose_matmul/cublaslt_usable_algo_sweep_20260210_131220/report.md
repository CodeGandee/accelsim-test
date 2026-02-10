
cuBLASLt usable-algo sweep (int8 square, row-major)
===================================================

# Run Metadata


`2026-02-10T13:22:45.598806+00:00`

Output: `reports/transpose_matmul/cublaslt_usable_algo_sweep_20260210_131220`

Sweep bin: `/data/ssd1/huangzhe/code/accelsim-test/cpp/build/Release/cublaslt_usable_algo_sweep` (pixi env: `cuda13`)

Workspace policy: `max_workspace_bytes=67108864`

Timing policy: `warmup=10`, `iters=50`
# Results (AB vs ABT_view)

## Executive Summary


- N=1000: best AB algo_id=64 (32.84us), best ABT_view algo_id=23 (14.40us) ⇒ ABT_view is ~2.28× faster (best-of-case).
  - algo_id=23: AB=NA (rank NA), ABT_view=14.40us (rank 1/2)
- N=1024: best AB algo_id=70 (8.59us), best ABT_view algo_id=70 (7.54us) ⇒ ABT_view is ~1.14× faster (best-of-case).
  - algo_id=23: AB=NA (rank NA), ABT_view=14.40us (rank 4/6)
- N=2048: best AB algo_id=71 (12.37us), best ABT_view algo_id=71 (12.40us) ⇒ ABT_view is ~1.00× faster (best-of-case).
  - algo_id=23: AB=NA (rank NA), ABT_view=45.13us (rank 4/6)
## N=1000


| algo_id | AB time_us | ABT_view time_us | ABT/AB |
|------:|----------:|---------------:|------:|
| 23 | NA | 14.40 | NA |
| 64 | 32.84 | 23.90 | 0.728 |

Full merged table: `merged_table.csv` (all algo_id rows).
## N=1024


| algo_id | AB time_us | ABT_view time_us | ABT/AB |
|------:|----------:|---------------:|------:|
| 70 | 8.59 | 7.54 | 0.878 |
| 71 | 8.84 | 7.67 | 0.868 |
| 21 | NA | 8.25 | NA |
| 23 | NA | 14.40 | NA |
| 20 | 28.77 | 26.69 | 0.928 |
| 64 | 38.96 | 32.85 | 0.843 |

Full merged table: `merged_table.csv` (all algo_id rows).
## N=2048


| algo_id | AB time_us | ABT_view time_us | ABT/AB |
|------:|----------:|---------------:|------:|
| 71 | 12.37 | 12.40 | 1.002 |
| 70 | 14.43 | 14.41 | 0.999 |
| 21 | NA | 26.70 | NA |
| 23 | NA | 45.13 | NA |
| 64 | 136.86 | 100.42 | 0.734 |
| 20 | 162.19 | 155.32 | 0.958 |

Full merged table: `merged_table.csv` (all algo_id rows).
