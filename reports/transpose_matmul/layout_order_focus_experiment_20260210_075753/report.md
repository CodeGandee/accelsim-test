
Layout-order focus experiment (N=1000 int8)
===========================================

# Run Metadata


`2026-02-10T07:58:23.865372+00:00`

Output: `tmp/layout_order_n1000_20260210_075753`

Repro: `cpp/build/Release/repro_algo23_int8_n1000` (pixi env: `cuda13`)
# Results (A/B order matrix; order_c=row)


| order_a | order_b | variant | time (us) | algo_id | tile | stages |
|--------:|--------:|---------|----------:|--------:|-----:|-------:|
| col | col | AB | 27.13 | 64 | 20 | 8 |
| col | col | ABT_view | 43.51 | 64 | 20 | 8 |
| col | col | ATB_view | 12.29 | 23 | 18 | 21 |
| col | row | AB | 40.91 | 64 | 20 | 8 |
| col | row | ABT_view | 27.13 | 64 | 20 | 8 |
| col | row | ATB_view | 32.79 | 64 | 20 | 8 |
| row | col | AB | 12.29 | 23 | 18 | 21 |
| row | col | ABT_view | 32.79 | 64 | 20 | 8 |
| row | col | ATB_view | 27.22 | 64 | 20 | 8 |
| row | row | AB | 32.79 | 64 | 20 | 8 |
| row | row | ABT_view | 12.29 | 23 | 18 | 21 |
| row | row | ATB_view | 40.92 | 64 | 20 | 8 |
# Output order check (baseline order_a=row, order_b=row)


| order_c | variant | time (us) | algo_id | tile | stages |
|--------:|---------|----------:|--------:|-----:|-------:|
| col | AB | 30.65 | 64 | 20 | 8 |
| col | ABT_view | 14.28 | 23 | 18 | 21 |
| col | ATB_view | 51.37 | 64 | 20 | 8 |
| row | AB | 32.79 | 64 | 20 | 8 |
| row | ABT_view | 12.29 | 23 | 18 | 21 |
| row | ATB_view | 40.92 | 64 | 20 | 8 |