"""Sweep + limiting-factor analysis. Writes results.json + results.md next to this file.

Per (N, E): fixed-action and masked-policy throughput (JAX unfused/scan, Warp unfused/graph), plus
  - mask-only throughput (cost of the legal-action mask alone)
  - launch overhead (unfused - fused, per step)
  - achieved memory traffic vs measured device copy bandwidth (roofline-style)
  - host<->device transfer of state[E,W]
  - memory (analytic live bytes; measured peak where the runtime exposes it)
and an automatic bottleneck label (launch / bandwidth / compute-or-latency).
"""
import json, os, statistics, time
import numpy as np
import jax, warp as wp
from ground import ground, make_pool
from env_jax import JaxEnv
from env_warp import WarpEnv

NS, ES, T = [4, 8, 16, 32], [1_000, 10_000, 100_000], 100
WARM, RUNS = 3, 10
HERE = os.path.dirname(os.path.abspath(__file__))
wp.init()
cuda = wp.is_cuda_available()
dev = "cuda:0" if cuda else "cpu"
label = "GPU" if cuda else "CPU"

def timeit(fn):
    for _ in range(WARM): fn()
    ts = []
    for _ in range(RUNS):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    q = statistics.quantiles(ts, n=4)
    return {"median": statistics.median(ts), "iqr": q[2] - q[0]}

@wp.kernel
def _copy(a: wp.array(dtype=wp.float32), b: wp.array(dtype=wp.float32)):
    i = wp.tid()
    b[i] = a[i]

def peak_bandwidth_gbs():
    """Measured device copy bandwidth (read+write), the reference for 'bandwidth-bound'."""
    n = 64_000_000 if cuda else 8_000_000
    a, b = wp.ones(n, dtype=wp.float32, device=dev), wp.zeros(n, dtype=wp.float32, device=dev)
    def f(): wp.synchronize(); wp.launch(_copy, dim=n, inputs=[a, b], device=dev); wp.synchronize()
    return 8 * n / timeit(f)["median"] / 1e9

def step_bytes(W):      # per env per step: state r+w, goal r, action, ridx, reward, done
    return 8 * W * 3 + 4 + 4 + 4 + 1

def jax_peak_mb():
    try: return jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 2**20
    except Exception: return None

def warp_peak_mb():
    try: return wp.get_mempool_used_mem_high(dev) / 2**20
    except Exception: return None

def classify(r, peak):
    tag = []
    for name, un, fu in (("jax", "jax_unfused", "jax_scan"), ("warp", "warp_unfused", "warp_graph")):
        if un in r and fu in r:
            ratio = r[un]["median"] / r[fu]["median"]
            if ratio > 1.5: tag.append(f"{name}: launch-bound unfused ({ratio:.1f}x fused)")
    gbs = r["warp_gbs"] if cuda else None
    frac = r["warp_gbs"] / peak
    fused = "warp_graph" if "warp_graph" in r else "warp_unfused"
    tag.append(f"warp fused: {'bandwidth-bound' if frac > 0.5 else 'compute/latency-bound'} ({100*frac:.0f}% of copy peak)")
    return "; ".join(tag)

peak = peak_bandwidth_gbs()
res = {"device": label, "jax": jax.__version__, "warp": wp.config.version,
       "jax_devices": [str(d) for d in jax.devices()], "T": T, "peak_copy_gbs": peak, "runs": []}
first = True
for N in NS:
    pre, add, dele = ground(N); ps, pg = make_pool(N)
    A, W = pre.shape
    jenv, wenv = JaxEnv(pre, add, dele, ps, pg), WarpEnv(pre, add, dele, ps, pg, dev)
    for E in ES:
        rng = np.random.default_rng(1)
        act = rng.integers(0, A, (T, E)).astype(np.int32)
        rid = rng.integers(0, len(ps), (T, E)).astype(np.int32)
        s0, g0 = ps[rid[0]], pg[rid[0]]
        row = {"N": N, "E": E, "W": W, "A": A}
        ja, jr = jax.device_put(act), jax.device_put(rid)
        js, jg = jax.device_put(s0), jax.device_put(g0)
        def jun():
            s, g = js, jg
            for t in range(T): s, g, r, d = jenv.step(s, g, ja[t], jr[t])
            jax.block_until_ready(s)
        def jsc(): jax.block_until_ready(jenv.rollout(js, jg, ja, jr))
        t0 = time.perf_counter(); jsc(); row["jax_compile_s"] = time.perf_counter() - t0
        row["jax_unfused"], row["jax_scan"] = timeit(jun), timeit(jsc)
        wenv.alloc(s0, g0)
        wa = [wp.array(a, dtype=wp.int32, device=dev) for a in act]
        wr = [wp.array(r, dtype=wp.int32, device=dev) for r in rid]
        if first:    # kernel build + module load, measured once (module is cached afterwards)
            wp.synchronize(); t0 = time.perf_counter(); wenv.step(wa[0], wr[0]); wp.synchronize()
            res["warp_first_call_s"] = time.perf_counter() - t0; first = False
        def wun():
            wp.synchronize()
            for t in range(T): wenv.step(wa[t], wr[t])
            wp.synchronize()
        row["warp_unfused"] = timeit(wun)
        if cuda:
            wenv.capture_rollout(wa, wr)
            def wg(): wp.synchronize(); wenv.replay(); wp.synchronize()
            row["warp_graph"] = timeit(wg)
        # masked policy (mask + sampling included)
        key = jax.random.PRNGKey(0)
        def jm(): jax.block_until_ready(jenv.rollout_masked(js, jg, key, jr))
        jm(); row["jax_masked_scan"] = timeit(jm)
        def wmun():
            wp.synchronize()
            for t in range(T): wenv.step_masked(wr[t], t)
            wp.synchronize()
        row["warp_masked_unfused"] = timeit(wmun)
        if cuda:
            wenv.capture_masked(wr)
            def wmg(): wp.synchronize(); wenv.replay(); wp.synchronize()
            row["warp_masked_graph"] = timeit(wmg)
        # mask alone (T calls, no step)
        def jmask():
            for _ in range(T): m = jenv.mask(js)
            jax.block_until_ready(m)
        def wmask():
            wp.synchronize()
            for _ in range(T): wenv.mask()
            wp.synchronize()
        row["jax_mask_only"], row["warp_mask_only"] = timeit(jmask), timeit(wmask)
        # host<->device transfer of state[E, W]
        def jxfer(): jax.block_until_ready(jax.device_put(np.asarray(js)))
        def wxfer(): wp.synchronize(); h = wenv.state.numpy().copy(); d = wp.array(h, dtype=wp.uint64, device=dev); wp.synchronize()
        row["jax_xfer"], row["warp_xfer"] = timeit(jxfer), timeit(wxfer)
        # memory
        row["live_mb"] = (E * W * 8 * 2 + T * E * 8) / 2**20      # state+goal+one action/ridx pair per step
        row["jax_peak_mb"], row["warp_peak_mb"] = jax_peak_mb(), warp_peak_mb()
        for k in ("jax_unfused", "jax_scan", "warp_unfused", "warp_graph",
                  "jax_masked_scan", "warp_masked_unfused", "warp_masked_graph",
                  "jax_mask_only", "warp_mask_only"):
            if k in row: row[k]["steps_per_s"] = E * T / row[k]["median"]
        fused = row["warp_graph"] if cuda else row["warp_unfused"]
        row["warp_gbs"] = step_bytes(W) * E * T / fused["median"] / 1e9
        row["limiter"] = classify(row, peak)
        res["runs"].append(row); print(row["N"], row["E"], row["limiter"], flush=True)

json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1)

def fmt(r, c): return f"{r[c]['steps_per_s']:.3g}" if c in r else "n/a"
with open(os.path.join(HERE, "results.md"), "w") as f:
    f.write(f"# Blocksworld JAX vs Warp ({label})\n\nJAX {res['jax']}, warp-lang {res['warp']}, devices {res['jax_devices']}\n")
    f.write(f"\nMeasured device copy bandwidth (reference peak): {peak:.1f} GB/s. Warp first-call (build + load): {res['warp_first_call_s']:.2f} s.\n")
    if not cuda: f.write("\n**No GPU: all numbers are CPU; no GPU conclusions.**\n")
    cols = ["jax_unfused", "jax_scan", "warp_unfused", "warp_graph",
            "jax_masked_scan", "warp_masked_unfused", "warp_masked_graph", "jax_mask_only", "warp_mask_only"]
    for N in NS:
        f.write(f"\n## N={N} throughput (env-steps/s)\n\n| E | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1) + "\n")
        for r in res["runs"]:
            if r["N"] == N: f.write(f"| {r['E']} | " + " | ".join(fmt(r, c) for c in cols) + " |\n")
    f.write("\n## Limiting factor\n\n| N | E | JAX launch ovh (us/step) | Warp launch ovh (us/step) | Warp GB/s | % copy peak | "
            "JAX xfer (ms) | Warp xfer (ms) | live MB | JAX peak MB | Warp peak MB | verdict |\n|" + "---|" * 12 + "\n")
    for r in res["runs"]:
        jo = (r["jax_unfused"]["median"] - r["jax_scan"]["median"]) / T * 1e6
        wo = (r["warp_unfused"]["median"] - r["warp_graph"]["median"]) / T * 1e6 if "warp_graph" in r else None
        n = lambda x, p=".1f": "n/a" if x is None else format(x, p)
        f.write(f"| {r['N']} | {r['E']} | {jo:.1f} | {n(wo)} | {r['warp_gbs']:.2f} | {100*r['warp_gbs']/peak:.0f}% | "
                f"{r['jax_xfer']['median']*1e3:.2f} | {r['warp_xfer']['median']*1e3:.2f} | {r['live_mb']:.1f} | "
                f"{n(r['jax_peak_mb'])} | {n(r['warp_peak_mb'])} | {r['limiter']} |\n")
    f.write("\nVerdict labels: *launch-bound* if unfused/fused > 1.5; *bandwidth-bound* if achieved traffic > 50% of the copy "
            "peak, else *compute/latency-bound*. Traffic counts state r/w, goal, action, reset index and outputs; the "
            "pre/add/del masks and pool are assumed cache-resident. JAX peak memory is process-cumulative.\n")
