"""Correctness gate: every env implementation must match ref_numpy exactly (state, goal, done counts, reward).
Run standalone (python check.py) or via verify() from bench_fair.py before any timing."""
import sys, numpy as np, jax, warp as wp
import ref_numpy as ref
from ground import ground, make_pool
from env_jax import JaxEnv
from env_jax_tuned import JaxTuned
from env_warp import WarpEnv
from env_warp_tuned import WarpTuned

def _eq(name, got, want):
    assert np.array_equal(np.asarray(got), np.asarray(want)), f"MISMATCH {name}"

def verify(N=4, E=40, T=40, seed=7, dev=None, verbose=True):
    wp.init()
    dev = dev or ("cuda:0" if wp.is_cuda_available() else "cpu")
    cuda = dev.startswith("cuda")
    pre, add, dele = ground(N); ps, pg = make_pool(N, K=256)
    rng = np.random.default_rng(seed)
    act = rng.integers(0, pre.shape[0], (T, E)).astype(np.int32)
    rid = rng.integers(0, len(ps), (T, E)).astype(np.int32)
    s0, g0 = ps[rid[0]], pg[rid[0]]

    # ---- reference: fixed actions
    rs, rg, nd = s0.copy(), g0.copy(), np.zeros(E, np.int32)
    for t in range(T):
        rs, rg, rew, d = ref.step(rs, rg, act[t], rid[t], pre, add, dele, ps, pg); nd += d
    fixed = (rs, rg, nd, rew)
    # ---- reference: greedy lookahead
    ls, lg, lnd = s0.copy(), g0.copy(), np.zeros(E, np.int32)
    for t in range(T):
        a = ref.lookahead_actions(ls, lg, pre, add, dele, seed, t)
        ls, lg, lrew, d = ref.step(ls, lg, a, rid[t], pre, add, dele, ps, pg); lnd += d
    look = (ls, lg, lnd, lrew)

    def cmp(tag, s, g, nd_, rew_, want):
        _eq(tag + " state", s, want[0]); _eq(tag + " goal", g, want[1])
        if nd_ is not None: _eq(tag + " ndone", nd_, want[2])
        _eq(tag + " reward", rew_, want[3])
        if verbose: print("  ok", tag, flush=True)

    # ---- JAX naive (fixed only)
    jn = JaxEnv(pre, add, dele, ps, pg)
    (s, g), (rews, ds) = jn.rollout(jax.device_put(s0), jax.device_put(g0), jax.device_put(act), jax.device_put(rid))
    cmp("jax_naive fixed", s, g, np.asarray(ds).sum(0), np.asarray(rews)[-1], fixed)
    # ---- JAX tuned
    for lay in ("aos", "soa"):
        jt = JaxTuned(pre, add, dele, ps, pg, lay)
        s, g, n_, r_ = jt.rollout(jt.to_layout(s0), jt.to_layout(g0), jax.device_put(act), jax.device_put(rid))
        cmp(f"jax_tuned_{lay} fixed", jt.to_aos(s), jt.to_aos(g), n_, r_, fixed)
        C = max(1, E // 4)
        while E % C: C -= 1
        s, g, n_, r_ = jt.look(jt.to_layout(s0), jt.to_layout(g0), jax.device_put(rid), np.uint32(seed), C)
        cmp(f"jax_tuned_{lay} look(C={C})", jt.to_aos(s), jt.to_aos(g), n_, r_, look)
    # ---- Warp naive (fixed only)
    wn = WarpEnv(pre, add, dele, ps, pg, dev); wn.alloc(s0, g0)
    wa = [wp.array(a, dtype=wp.int32, device=dev) for a in act]; wr = [wp.array(r, dtype=wp.int32, device=dev) for r in rid]
    for t in range(T): wn.step(wa[t], wr[t])
    cmp("warp_naive fixed", wn.state.numpy(), wn.goal.numpy(), None, wn.reward.numpy(), fixed)
    # ---- Warp tuned
    for lay in ("aos", "soa"):
        wt = WarpTuned(pre, add, dele, ps, pg, dev, lay)
        wt.alloc(s0, g0)
        for t in range(T): wt.step(wa[t], wr[t])
        cmp(f"warp_tuned_{lay} fixed", wt.state_aos(), np.ascontiguousarray(wt.goal.numpy().T), wt.ndone.numpy(), wt.reward.numpy(), fixed)
        wt.alloc(s0, g0)
        for t in range(T): wt.step_look(wr[t], seed, t)
        cmp(f"warp_tuned_{lay} look", wt.state_aos(), np.ascontiguousarray(wt.goal.numpy().T), wt.ndone.numpy(), wt.reward.numpy(), look)
        if cuda:
            wt.alloc(s0, g0); wt.capture_fixed(wa, wr); wt.replay(); wp.synchronize()
            cmp(f"warp_tuned_{lay} fixed graph", wt.state_aos(), np.ascontiguousarray(wt.goal.numpy().T), wt.ndone.numpy(), wt.reward.numpy(), fixed)
            wt.alloc(s0, g0); wt.capture_look(wr, seed); wt.replay(); wp.synchronize()
            cmp(f"warp_tuned_{lay} look graph", wt.state_aos(), np.ascontiguousarray(wt.goal.numpy().T), wt.ndone.numpy(), wt.reward.numpy(), look)
    return True

if __name__ == "__main__":
    for N, E, T in [(4, 40, 40), (8, 24, 30), (16, 12, 20)]:     # N=8 -> W=2, N=16 -> W=5 (multi-word)
        print(f"verify N={N} E={E} T={T}", flush=True)
        verify(N, E, T)
    print("ALL OK")
