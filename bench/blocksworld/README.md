# Blocksworld: JAX vs NVIDIA Warp

How fast can we step thousands of parallel planning environments on a GPU, and does the framework matter?

PDDLGym is a single-process CPU env, so it can't scale to thousands of envs. The approach here is to **ground the domain offline** and step **bitset states** on the GPU. This benchmark builds the same grounded Blocksworld env in JAX and in Warp and compares them.

## The question

- **JAX/XLA** runs the same dense code for every env in the batch.
- **Warp** runs one thread per env, so each env can skip work.

For a small dense bitset step the two may perform about the same. Warp is more likely to win at large N with few legal actions, or with per-env divergence. This is an expectation to test, not assume.

## The env

Standard 4-operator STRIPS Blocksworld (`pickup`, `putdown`, `stack`, `unstack`) with N blocks.

| N | Facts `(N+1)²` | uint64 words/state | Ground actions `2N²` |
|---|---|---|---|
| 4 | 25 | 1 | 32 |
| 8 | 81 | 2 | 128 |
| 16 | 289 | 5 | 512 |
| 32 | 1089 | 18 | 2048 |

- **State:** a bitset, `uint64[W]`. Each ground action has `pre`, `add`, `del` masks.
- **Applicable:** `(s & pre) == pre`. **Next state:** `(s & ~del) | add`. Inapplicable actions are no-ops.
- **Goal:** a mask `g` over the `on` facts; done when `(s & g) == g`. Reward 1.0 on done, else 0.0.
- **Auto-reset:** finished envs are reset from a pool of 4096 pre-generated (state, goal) pairs, so no sampling or grounding runs on the GPU.

## Layout

| File | Role |
|---|---|
| `ground.py` | Grounder, masks, problem pool (shared by both envs) |
| `ref_numpy.py` | Slow, obviously-correct reference step (python-int bitsets) |
| `env_jax.py` | JAX env: `jit`, `vmap`-style batching, `lax.scan` rollouts |
| `env_warp.py` | Warp env: mask/step/sample kernels, CUDA-graph capture |
| `bench.py` | Sweep, writes `results.json` and `results.md` |

## Run

```bash
pip install -e .
python bench.py
```

Sweep: N ∈ {4, 8, 16, 32}, E ∈ {1k, 10k, 100k}, T = 100 steps, 3 warmups + 10 timed runs (median and IQR). Without a GPU everything runs on CPU, is labeled as such, and says nothing about GPU performance.

## What is measured

Two action modes, both in env-steps per second (`E*T / time`):

1. **Fixed random actions:** pre-generated on device, so RNG cost is excluded.
2. **Masked random policy:** uniform over legal actions each step, so mask cost is included.

Four execution styles:

| Backend | Unfused (one call per step) | Fused (one call per T steps) |
|---|---|---|
| JAX | `jit` step in a Python loop | `lax.scan` |
| Warp | kernel launches in a Python loop | CUDA graph (GPU only) |

The unfused-vs-fused gap is the launch overhead. Fixed-vs-masked is the mask cost. JAX compile time is recorded separately.

## Caveats

- The Warp masked policy is a **fused** mask+sample kernel and never materializes the `[E, A]` mask. The JAX version builds the full mask. This favors Warp in the masked mode.
- The Warp CUDA-graph path is untested without a GPU.
- Not yet measured: Warp compile time, host-device transfer, peak memory, 1M envs.
- No equivalence test file yet. A throwaway check showed the reference, JAX and Warp agree on states and dones for N=4, 50 envs, 300 steps.
- Goals only constrain `on` facts of a random target arrangement.
