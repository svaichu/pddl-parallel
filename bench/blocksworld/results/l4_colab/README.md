# Fair JAX vs Warp sweep: NVIDIA L4 (Colab VM)

Commit under test: f06fdfa. jax/jaxlib 0.12.1 (cuda12), warp-lang 1.19.0, T=100, N in {4,8,16,32}, E in {1k,10k,100k}.
Correctness gate (`check.verify`) passed on GPU before timing. Raw medians/IQR: `results_N*.json`.
Throughput table (M steps/s, all variants): `summary_Msteps_per_s.csv` (F: = fixed-action rollouts, L: = greedy lookahead).

Note: single run on an L4; absolute numbers will differ on other GPUs (the hpc-alt H100 runs were queued separately).
