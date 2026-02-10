# Layout-order focus experiment (N=1000 int8)

This repo has an observed `N=1000` int8 GEMM discontinuity where `ABT_view` can select a much faster cuBLASLt kernel than `AB`. This experiment is a focused follow-up that varies **A/B storage order independently** (row/col, including mixed cases) to test the hypothesis that **K-contiguity / layout order** drives kernel selection and performance.

## What it runs

Transpose-view variants (view-only; no copy/pack cases):

- `AB`: `trans_a=N`, `trans_b=N`
- `ATB_view`: `trans_a=T`, `trans_b=N`
- `ABT_view`: `trans_a=N`, `trans_b=T`

Orders:

- Full A/B order matrix (C fixed to row-major):
  - `order_a ∈ {row,col}` × `order_b ∈ {row,col}` × variants (12 runs)
- Limited output-order sensitivity check:
  - `order_a=row, order_b=row` and `order_c ∈ {row,col}` × variants
  - In practice, the matrix already includes `order_c=row`; the script adds only the `order_c=col` runs.

Optional controls:

- `--symmetric-inputs`: generate symmetric A and B (`A=Aᵀ`, `B=Bᵀ`) so `AB`, `ATB_view`, `ABT_view` are “same math”.
- `--nsys` / `--ncu`: capture kernel evidence under `<out_dir>/profiles/<case_id>/...`.

## Build the repro binary

```bash
cd cpp
conan profile detect --force
conan install . -b missing
cmake --preset conan-release
cmake --build --preset conan-release -j
```

The binary used by the experiment is:

- `cpp/build/Release/repro_algo23_int8_n1000`

## Run the experiment

Minimal (just timing + selected algo/config):

```bash
pixi run python scripts/layout_order_focus_experiment.py \
  --out-dir tmp/layout_order_n1000 \
  --repro-bin ./cpp/build/Release/repro_algo23_int8_n1000 \
  --pixi-env cuda13
```

Same-math mode (recommended when comparing across transpose-view variants):

```bash
pixi run python scripts/layout_order_focus_experiment.py \
  --out-dir tmp/layout_order_n1000_sym \
  --repro-bin ./cpp/build/Release/repro_algo23_int8_n1000 \
  --pixi-env cuda13 \
  --symmetric-inputs
```

With kernel discovery + profiling per case:

```bash
pixi run python scripts/layout_order_focus_experiment.py \
  --out-dir tmp/layout_order_n1000_profiles \
  --repro-bin ./cpp/build/Release/repro_algo23_int8_n1000 \
  --pixi-env cuda13 \
  --symmetric-inputs \
  --nsys \
  --ncu \
  --ncu-set basic \
  --profile-iters 1 --profile-warmup 0
```

## Outputs

Under the chosen `--out-dir`:

- `meta.json`: run metadata (host + git + args)
- `results.json` / `results.csv`: per-case index (order_a/order_b/order_c/variant/algo/timing)
- `report.md`: concise summary tables
- `cases/<case_id>/stdout.txt`: raw repro output for each case
- `profiles/<case_id>/...`: optional `nsys` / `ncu` artifacts (when enabled)

## How to interpret

- **Winner flips across A/B orders**:
  - If the contiguity hypothesis holds, the “fast path” (algo/kernel family) may move between `ABT_view` and `ATB_view` as you change `order_a`/`order_b`.
  - Look for correlated shifts in `algo_id`/tile/stages and (when captured) the main kernel name.

- **Mixed-order cases (`row/col`, `col/row`)**:
  - These are often the most diagnostic: they separate “A matters” vs “B matters” effects.

- **Output-order sensitivity (`order_c`)**:
  - The limited `order_c` sweep (only for baseline `order_a=row, order_b=row`) is a sanity check.
  - If `order_c` changes selection/timing materially, include it explicitly in comparisons (and consider expanding the sweep).

