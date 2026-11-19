"""Minimal sweep: fixed random actions, JAX unfused/scan vs Warp unfused/graph. Writes results.json + results.md."""
import json, statistics, time
import numpy as np
import jax, warp as wp
from ground import ground, make_pool
from env_jax import JaxEnv
from env_warp import WarpEnv

NS, ES, T = [4, 8, 16, 32], [1_000, 10_000, 100_000], 100
WARM, RUNS = 3, 10
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

res = {"device": label, "jax": jax.__version__, "warp": wp.config.version,
       "jax_devices": [str(d) for d in jax.devices()], "T": T, "runs": []}
for N in NS:
    pre, add, dele = ground(N); ps, pg = make_pool(N)
    A = pre.shape[0]
    jenv, wenv = JaxEnv(pre, add, dele, ps, pg), WarpEnv(pre, add, dele, ps, pg, dev)
    for E in ES:
        rng = np.random.default_rng(1)
        act = rng.integers(0, A, (T, E)).astype(np.int32)
        rid = rng.integers(0, len(ps), (T, E)).astype(np.int32)
        s0, g0 = ps[rid[0]], pg[rid[0]]
        row = {"N": N, "E": E}
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
        def wun():
            wp.synchronize()
            for t in range(T): wenv.step(wa[t], wr[t])
            wp.synchronize()
        row["warp_unfused"] = timeit(wun)
        if cuda:
            wenv.capture_rollout(wa, wr)
            def wg(): wp.synchronize(); wenv.replay(); wp.synchronize()
            row["warp_graph"] = timeit(wg)
        # masked random policy (includes mask + sampling cost)
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
        for k in ("jax_unfused", "jax_scan", "warp_unfused", "warp_graph",
                  "jax_masked_scan", "warp_masked_unfused", "warp_masked_graph"):
            if k in row: row[k]["steps_per_s"] = E * T / row[k]["median"]
        res["runs"].append(row); print(row, flush=True)

json.dump(res, open("results.json", "w"), indent=1)
with open("results.md", "w") as f:
    f.write(f"# Blocksworld JAX vs Warp ({label})\n\nJAX {res['jax']}, warp-lang {res['warp']}, devices {res['jax_devices']}\n")
    if not cuda: f.write("\n**No GPU: all numbers are CPU; no GPU conclusions.**\n")
    cols = ["jax_unfused", "jax_scan", "warp_unfused", "warp_graph",
            "jax_masked_scan", "warp_masked_unfused", "warp_masked_graph"]
    for N in NS:
        f.write(f"\n## N={N} (env-steps/s)\n\n| E | " + " | ".join(cols) + " |\n|" + "---|" * 5 + "\n")
        for r in res["runs"]:
            if r["N"] == N:
                f.write(f"| {r['E']} | " + " | ".join(f"{r[c]['steps_per_s']:.3g}" if c in r else "n/a" for c in cols) + " |\n")
