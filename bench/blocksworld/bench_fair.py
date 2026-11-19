"""Fair JAX-vs-Warp benchmark. Per (N, E):
  fixed-action rollouts : jax naive (unfused/scan), jax tuned (aos/soa), warp naive (unfused/graph), warp tuned (aos/soa; unfused/graph)
  greedy lookahead      : jax tuned (aos/soa, chunked), warp tuned (aos/soa, graph)
All 'tuned' variants return identical outputs (final state/goal, last reward, done counter).
Usage: python bench_fair.py --N 4 8 --E 1000 10000 --T 100 --out results [--cpu-ok]
Each invocation should cover one N (run separate processes per N so memory stats don't accumulate)."""
import argparse, json, os, statistics, time
import numpy as np, jax, warp as wp
from common import require_same_gpu, env_info
from ground import ground, make_pool
from env_jax import JaxEnv
from env_jax_tuned import JaxTuned
from env_warp import WarpEnv
from env_warp_tuned import WarpTuned
from check import verify

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, nargs="+", default=[4, 8, 16, 32])
ap.add_argument("--E", type=int, nargs="+", default=[1000, 10000, 100000])
ap.add_argument("--T", type=int, default=100)
ap.add_argument("--warm", type=int, default=3)
ap.add_argument("--runs", type=int, default=10)
ap.add_argument("--out", default="results")
ap.add_argument("--cpu-ok", action="store_true", help="allow CPU run (smoke test; numbers say nothing about GPU)")
ap.add_argument("--no-verify", action="store_true")
args = ap.parse_args()

wp.init()
if not args.cpu_ok:
    require_same_gpu()
cuda = wp.is_cuda_available()
dev = "cuda:0" if cuda else "cpu"
T, SEED = args.T, 123

def timeit(fn):
    for _ in range(args.warm): fn()
    ts = []
    for _ in range(args.runs):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    q = statistics.quantiles(ts, n=4)
    return {"median": statistics.median(ts), "iqr": q[2] - q[0]}

def sync(): wp.synchronize()

os.makedirs(args.out, exist_ok=True)
for N in args.N:
    if not args.no_verify:
        print(f"[verify] N={N}", flush=True); verify(N, 24, 20, dev=dev, verbose=False)
    pre, add, dele = ground(N); ps, pg = make_pool(N)
    A, W = pre.shape
    jn = JaxEnv(pre, add, dele, ps, pg)
    jt = {l: JaxTuned(pre, add, dele, ps, pg, l) for l in ("aos", "soa")}
    wn = WarpEnv(pre, add, dele, ps, pg, dev)
    wt = {l: WarpTuned(pre, add, dele, ps, pg, dev, l) for l in ("aos", "soa")}
    res = {"N": N, "W": W, "A": A, "T": T, "device": "GPU" if cuda else "CPU", "info": env_info(), "rows": []}
    for E in args.E:
        rng = np.random.default_rng(1)
        act = rng.integers(0, A, (T, E)).astype(np.int32)
        rid = rng.integers(0, len(ps), (T, E)).astype(np.int32)
        s0, g0 = ps[rid[0]], pg[rid[0]]
        row = {"E": E}
        ja, jr = jax.device_put(act), jax.device_put(rid)
        ja_l, jr_l = [ja[t] for t in range(T)], [jr[t] for t in range(T)]     # pre-sliced: no eager indexing in timed loop
        wa = [wp.array(a, dtype=wp.int32, device=dev) for a in act]
        wr = [wp.array(r, dtype=wp.int32, device=dev) for r in rid]
        jax.block_until_ready((ja_l, jr_l)); sync()

        # ---------------- fixed actions ----------------
        js, jg = jax.device_put(s0), jax.device_put(g0)
        def jun():
            s, g = js, jg
            for t in range(T): s, g, r, d = jn.step(s, g, ja_l[t], jr_l[t])
            jax.block_until_ready(s)
        def jsc(): jax.block_until_ready(jn.rollout(js, jg, ja, jr))
        t0 = time.perf_counter(); jsc(); row["jax_naive_compile_s"] = time.perf_counter() - t0
        row["fixed_jax_naive_unfused"], row["fixed_jax_naive_scan"] = timeit(jun), timeit(jsc)

        for l, env in jt.items():
            st = [env.to_layout(s0), env.to_layout(g0)]
            def f(env=env, st=st):
                out = env.rollout(st[0], st[1], ja, jr); st[0], st[1] = out[0], out[1]; jax.block_until_ready(out)
            t0 = time.perf_counter(); f(); row[f"jax_tuned_{l}_compile_s"] = time.perf_counter() - t0
            row[f"fixed_jax_tuned_{l}"] = timeit(f)
            st = [env.to_layout(s0), env.to_layout(g0)]
            out = env.rollout(st[0], st[1], ja, jr); row[f"done_rate_fixed"] = float(np.asarray(out[2]).sum() / (E * T))

        wn.alloc(s0, g0)
        def wun():
            sync()
            for t in range(T): wn.step(wa[t], wr[t])
            sync()
        row["fixed_warp_naive_unfused"] = timeit(wun)
        if cuda:
            wn.capture_rollout(wa, wr)
            def wg(): sync(); wn.replay(); sync()
            row["fixed_warp_naive_graph"] = timeit(wg)
        for l, env in wt.items():
            env.alloc(s0, g0)
            def wu(env=env):
                sync()
                for t in range(T): env.step(wa[t], wr[t])
                sync()
            row[f"fixed_warp_tuned_{l}_unfused"] = timeit(wu)
            if cuda:
                env.capture_fixed(wa, wr)
                def wgr(env=env): sync(); env.replay(); sync()
                row[f"fixed_warp_tuned_{l}_graph"] = timeit(wgr)

        # ---------------- greedy lookahead ----------------
        for l, env in jt.items():
            C = env.chunk_for(E); row["jax_chunk"] = C
            st = [env.to_layout(s0), env.to_layout(g0)]
            def f(env=env, st=st, C=C):
                out = env.look(st[0], st[1], jr, np.uint32(SEED), C); st[0], st[1] = out[0], out[1]; jax.block_until_ready(out)
            t0 = time.perf_counter(); f(); row[f"jax_look_{l}_compile_s"] = time.perf_counter() - t0
            row[f"look_jax_tuned_{l}"] = timeit(f)
            st = [env.to_layout(s0), env.to_layout(g0)]
            out = env.look(st[0], st[1], jr, np.uint32(SEED), C); row["done_rate_look"] = float(np.asarray(out[2]).sum() / (E * T))
        for l, env in wt.items():
            env.alloc(s0, g0)
            if cuda:
                env.capture_look(wr, SEED)
                def wgl(env=env): sync(); env.replay(); sync()
                row[f"look_warp_tuned_{l}_graph"] = timeit(wgl)
            else:
                def wul(env=env):
                    sync()
                    for t in range(T): env.step_look(wr[t], SEED, t)
                    sync()
                row[f"look_warp_tuned_{l}_unfused"] = timeit(wul)

        for k, v in row.items():
            if isinstance(v, dict):
                v["steps_per_s"] = E * T / v["median"]
                if k.startswith("look_"): v["children_per_s"] = E * T * A / v["median"]
        res["rows"].append(row)
        print(f"N={N} E={E} done", flush=True)
        json.dump(res, open(os.path.join(args.out, f"results_N{N}.json"), "w"), indent=1)
