# JAX greedy-lookahead bottleneck at N=32 (NVIDIA L4, commit 596a38c/19b4df8)

## Question
Why does tuned-JAX lookahead plateau at ~2.3-2.6 M steps/s (~5 G children/s) at N=32 (A=2048, W=18), independent of E, and fall ~10x from N=16, while Warp reaches 13.8 M steps/s (28 G children/s)?

## Findings
1. Not a serial/per-call bottleneck (earlier guess retracted). Throughput is flat in E because the kernel is already saturated.
2. Chunk size is not the lever: C=25..10000 gives 1.5 -> 6.1 G children/s; C>=1000 saturates (5.7-6.1 G/s), including an unchunked pass (2.8 GiB [C,A,W] logical tensor).
3. nsys (--cuda-graph-trace=node): 2 fusions = ~96% of GPU time: input_reduce_fusion (72%, ~254 us/chunk at C=1000) and loop_reduce_fusion (24%, ~83 us).
4. ncu on input_reduce_fusion (N=32, C=1000): DRAM 2.4% of peak (successor tensor is NOT materialized), L1/tex 88.6%, L2 80.2%, SM throughput 16.5%, warps active 16.7%. The kernel is bound by L1/L2 bandwidth, not arithmetic or HBM.
5. Likely cause: each (env, action, word) element re-reads pre/add/dele[a,w] (3 x 8 B). At 5.5 G children/s x 18 words that is ~2.4 TB/s of cache traffic. The Warp kernel (one thread per env, loop over A and W) reads those table values at warp-uniform addresses, so one load serves 32 envs.
6. Source-level JAX rewrites did not help (identical selected actions in all cases):
   - unrolling over W (elementwise on [C,A]): 4.7 vs 5.5 G children/s (N=32, C=1000)
   - env-minor layout [A,W,C]: 3.1-4.9 vs 4.8-5.9 G children/s
   XLA chooses its own thread mapping inside the fusion, so the access pattern cannot be controlled from jnp.

## Caveats
- ncu duration (601 us) is longer than nsys (254 us) because ncu locks clocks and serializes; use only the percentages.
- look_variants.py times one jitted step per call; for N<=16 it is dispatch-bound (~100 us) and only the N=32 comparison is meaningful.
- Single L4; bandwidth ceilings differ on other GPUs.

## Suggested fix (not done)
Replace the jnp lookahead with a custom kernel: Warp via warp.jax_experimental (jax_kernel/jax_callable) or a Pallas/Triton kernel with one thread per env and warp-uniform table loads. This reproduces what the Warp env already does.

## Files
chunk_sweep.py (chunk-size sweep), look_variants.py (variant check vs reference argmin).
