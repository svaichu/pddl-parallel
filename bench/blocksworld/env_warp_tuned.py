"""Tuned Warp env. Differences from env_warp.WarpEnv (the naive baseline):
  - W (words/state) and A (actions) are compile-time constants -> inner loops unroll
  - kernels index state[w, e]; layout "soa" stores [W,E] contiguous (coalesced across threads),
    "aos" stores [E,W] and passes a transposed *view* (same kernel, strided access)
  - per-env done counter, same outputs as JaxTuned (final reward + ndone)
  - fused greedy one-step lookahead + step kernel (no [E,A] or [E,A,W] tensor ever materialized)
  - CUDA-graph warmup runs on clones, so a captured graph applies exactly T steps
"""
import numpy as np, warp as wp
from common import C1, C2, C3, C4, F1, F2
wp.config.quiet = True
wp.config.max_unroll = 32

@wp.func
def hash32(seed: wp.uint32, t: wp.uint32, e: wp.uint32, a: wp.uint32) -> wp.uint32:
    x = (seed * wp.uint32(C1)) ^ (t * wp.uint32(C2)) ^ (e * wp.uint32(C3)) ^ (a * wp.uint32(C4))
    x = x ^ (x >> wp.uint32(16))
    x = x * wp.uint32(F1)
    x = x ^ (x >> wp.uint32(13))
    x = x * wp.uint32(F2)
    x = x ^ (x >> wp.uint32(16))
    return x

@wp.func
def popcount64(x: wp.uint64) -> int:
    x = x - ((x >> wp.uint64(1)) & wp.uint64(6148914691236517205))
    x = (x & wp.uint64(3689348814741910323)) + ((x >> wp.uint64(2)) & wp.uint64(3689348814741910323))
    x = (x + (x >> wp.uint64(4))) & wp.uint64(1085102592571150095)
    return int((x * wp.uint64(72340172838076673)) >> wp.uint64(56))

_cache = {}
def kernels(W, A):
    if (W, A) in _cache:
        return _cache[(W, A)]

    @wp.kernel(module="unique")
    def step_k(state: wp.array2d(dtype=wp.uint64), goal: wp.array2d(dtype=wp.uint64),
               action: wp.array(dtype=wp.int32), ridx: wp.array(dtype=wp.int32),
               pre: wp.array2d(dtype=wp.uint64), add: wp.array2d(dtype=wp.uint64),
               dele: wp.array2d(dtype=wp.uint64), ps: wp.array2d(dtype=wp.uint64),
               pg: wp.array2d(dtype=wp.uint64), reward: wp.array(dtype=wp.float32),
               ndone: wp.array(dtype=wp.int32)):
        e = wp.tid()
        a = action[e]
        app = bool(True)
        for w in range(W):
            if (state[w, e] & pre[a, w]) != pre[a, w]:
                app = False
        fin = bool(True)
        for w in range(W):
            s = state[w, e]
            if app:
                s = (s & ~dele[a, w]) | add[a, w]
                state[w, e] = s
            if (s & goal[w, e]) != goal[w, e]:
                fin = False
        reward[e] = wp.where(fin, 1.0, 0.0)
        if fin:
            ndone[e] = ndone[e] + 1
            r = ridx[e]
            for w in range(W):
                state[w, e] = ps[r, w]
                goal[w, e] = pg[r, w]

    @wp.kernel(module="unique")
    def look_k(state: wp.array2d(dtype=wp.uint64), goal: wp.array2d(dtype=wp.uint64),
               ridx: wp.array(dtype=wp.int32),
               pre: wp.array2d(dtype=wp.uint64), add: wp.array2d(dtype=wp.uint64),
               dele: wp.array2d(dtype=wp.uint64), ps: wp.array2d(dtype=wp.uint64),
               pg: wp.array2d(dtype=wp.uint64), reward: wp.array(dtype=wp.float32),
               ndone: wp.array(dtype=wp.int32), seed: wp.uint32, t: wp.uint32):
        e = wp.tid()
        best = ~wp.uint32(0)
        a = int(0)
        for b in range(A):
            ok = bool(True)
            h = int(0)
            for w in range(W):
                s0 = state[w, e]
                if (s0 & pre[b, w]) != pre[b, w]:
                    ok = False
                ch = (s0 & ~dele[b, w]) | add[b, w]
                h += popcount64(goal[w, e] & ~ch)
            if ok:
                key = (wp.uint32(h) << wp.uint32(20)) | (hash32(seed, t, wp.uint32(e), wp.uint32(b)) >> wp.uint32(12))
                if key < best:
                    best = key
                    a = b
        fin = bool(True)
        for w in range(W):
            s = (state[w, e] & ~dele[a, w]) | add[a, w]      # chosen action is always applicable
            state[w, e] = s
            if (s & goal[w, e]) != goal[w, e]:
                fin = False
        reward[e] = wp.where(fin, 1.0, 0.0)
        if fin:
            ndone[e] = ndone[e] + 1
            r = ridx[e]
            for w in range(W):
                state[w, e] = ps[r, w]
                goal[w, e] = pg[r, w]

    _cache[(W, A)] = (step_k, look_k)
    return _cache[(W, A)]

class WarpTuned:
    def __init__(self, pre, add, dele, pool_s, pool_g, device=None, layout="soa"):
        assert layout in ("aos", "soa")
        self.dev, self.layout, self.soa = device, layout, layout == "soa"
        mk = lambda x: wp.array(np.ascontiguousarray(x), dtype=wp.uint64, device=device)
        self.pre, self.add, self.dele, self.ps, self.pg = map(mk, (pre, add, dele, pool_s, pool_g))
        self.A, self.W = pre.shape
        self.step_k, self.look_k = kernels(self.W, self.A)
        self.graph = None

    def _arr(self, x):                                  # x: numpy [E,W] -> logical [W,E] device array
        if self.soa:
            return wp.array(np.ascontiguousarray(x.T), dtype=wp.uint64, device=self.dev)
        self._keep = getattr(self, "_keep", [])
        st = wp.array(np.ascontiguousarray(x), dtype=wp.uint64, device=self.dev)
        self._keep.append(st)
        return st.transpose()

    def alloc(self, state, goal):
        E = state.shape[0]
        self.E = E
        self._keep = []
        self.state, self.goal = self._arr(state), self._arr(goal)
        self.reward = wp.zeros(E, dtype=wp.float32, device=self.dev)
        self.ndone = wp.zeros(E, dtype=wp.int32, device=self.dev)
        self.graph = None

    def state_aos(self):
        return np.ascontiguousarray(self.state.numpy().T)

    def _step(self, state, goal, reward, ndone, action, ridx):
        wp.launch(self.step_k, dim=self.E, device=self.dev,
                  inputs=[state, goal, action, ridx, self.pre, self.add, self.dele, self.ps, self.pg, reward, ndone])

    def _look(self, state, goal, reward, ndone, ridx, seed, t):
        wp.launch(self.look_k, dim=self.E, device=self.dev,
                  inputs=[state, goal, ridx, self.pre, self.add, self.dele, self.ps, self.pg, reward, ndone,
                          wp.uint32(seed), wp.uint32(t)])

    def step(self, action, ridx):
        self._step(self.state, self.goal, self.reward, self.ndone, action, ridx)

    def step_look(self, ridx, seed, t):
        self._look(self.state, self.goal, self.reward, self.ndone, ridx, seed, t)

    def _clones(self):
        return (wp.clone(self.state), wp.clone(self.goal), wp.clone(self.reward), wp.clone(self.ndone))

    def capture_fixed(self, actions, ridxs):
        self._step(*self._clones()[:4], actions[0], ridxs[0])      # module load on scratch copies
        wp.synchronize()
        with wp.ScopedCapture(device=self.dev) as cap:
            for a, r in zip(actions, ridxs):
                self.step(a, r)
        self.graph = cap.graph

    def capture_look(self, ridxs, seed):
        self._look(*self._clones()[:4], ridxs[0], seed, 0)
        wp.synchronize()
        with wp.ScopedCapture(device=self.dev) as cap:
            for t, r in enumerate(ridxs):
                self.step_look(r, seed, t)
        self.graph = cap.graph

    def replay(self):
        wp.capture_launch(self.graph)
