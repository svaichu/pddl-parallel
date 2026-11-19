import numpy as np, warp as wp
wp.config.quiet = True

@wp.kernel
def mask_kernel(state: wp.array2d(dtype=wp.uint64), pre: wp.array2d(dtype=wp.uint64),
                out: wp.array2d(dtype=wp.bool), W: int):
    e, a = wp.tid()
    ok = bool(True)
    for w in range(W):
        if (state[e, w] & pre[a, w]) != pre[a, w]:
            ok = False
    out[e, a] = ok

@wp.kernel
def sample_kernel(state: wp.array2d(dtype=wp.uint64), pre: wp.array2d(dtype=wp.uint64),
                  action: wp.array(dtype=wp.int32), A: int, W: int, seed: int):
    """Uniform sample among legal actions (fused mask + sample, no [E,A] mask materialized)."""
    e = wp.tid()
    cnt = int(0)
    for a in range(A):
        ok = bool(True)
        for w in range(W):
            if (state[e, w] & pre[a, w]) != pre[a, w]:
                ok = False
        if ok:
            cnt += 1
    rng = wp.rand_init(seed, e)
    k = wp.min(int(wp.randf(rng) * float(cnt)), cnt - 1)
    i = int(0)
    for a in range(A):
        ok = bool(True)
        for w in range(W):
            if (state[e, w] & pre[a, w]) != pre[a, w]:
                ok = False
        if ok:
            if i == k:
                action[e] = a
            i += 1

@wp.kernel
def step_kernel(state: wp.array2d(dtype=wp.uint64), goal: wp.array2d(dtype=wp.uint64),
                action: wp.array(dtype=wp.int32), ridx: wp.array(dtype=wp.int32),
                pre: wp.array2d(dtype=wp.uint64), add: wp.array2d(dtype=wp.uint64),
                dele: wp.array2d(dtype=wp.uint64), ps: wp.array2d(dtype=wp.uint64),
                pg: wp.array2d(dtype=wp.uint64), reward: wp.array(dtype=wp.float32),
                done: wp.array(dtype=wp.bool), W: int):
    e = wp.tid()
    a = action[e]
    app = bool(True)
    for w in range(W):
        if (state[e, w] & pre[a, w]) != pre[a, w]:
            app = False
    fin = bool(True)
    for w in range(W):
        s = state[e, w]
        if app:
            s = (s & ~dele[a, w]) | add[a, w]
            state[e, w] = s
        if (s & goal[e, w]) != goal[e, w]:
            fin = False
    done[e] = fin
    reward[e] = wp.where(fin, 1.0, 0.0)
    if fin:
        r = ridx[e]
        for w in range(W):
            state[e, w] = ps[r, w]
            goal[e, w] = pg[r, w]

class WarpEnv:
    def __init__(self, pre, add, dele, pool_s, pool_g, device=None):
        self.dev = device
        mk = lambda x: wp.array(x, dtype=wp.uint64, device=device)
        self.pre, self.add, self.dele, self.ps, self.pg = map(mk, (pre, add, dele, pool_s, pool_g))
        self.A, self.W = pre.shape
        self.K = pool_s.shape[0]
        self.graph = None
        self.act = None

    def alloc(self, state, goal):                      # in-place envs; state/goal numpy [E,W]
        E = state.shape[0]
        self.state = wp.array(state, dtype=wp.uint64, device=self.dev)
        self.goal = wp.array(goal, dtype=wp.uint64, device=self.dev)
        self.reward = wp.zeros(E, dtype=wp.float32, device=self.dev)
        self.done = wp.zeros(E, dtype=wp.bool, device=self.dev)

    def mask(self):
        out = wp.zeros((self.state.shape[0], self.A), dtype=wp.bool, device=self.dev)
        wp.launch(mask_kernel, dim=out.shape, inputs=[self.state, self.pre, out, self.W], device=self.dev)
        return out

    def step(self, action, ridx):                      # int32 device arrays [E]
        wp.launch(step_kernel, dim=self.state.shape[0], device=self.dev,
                  inputs=[self.state, self.goal, action, ridx, self.pre, self.add, self.dele,
                          self.ps, self.pg, self.reward, self.done, self.W])

    def capture_rollout(self, actions, ridxs):         # lists of [E] arrays; one graph of T steps
        self.step(actions[0], ridxs[0])                # warmup (module load)
        with wp.ScopedCapture(device=self.dev) as cap:
            for a, r in zip(actions, ridxs):
                self.step(a, r)
        self.graph = cap.graph

    def step_masked(self, ridx, seed):
        if self.act is None:
            self.act = wp.zeros(self.state.shape[0], dtype=wp.int32, device=self.dev)
        wp.launch(sample_kernel, dim=self.state.shape[0], device=self.dev,
                  inputs=[self.state, self.pre, self.act, self.A, self.W, seed])
        self.step(self.act, ridx)

    def capture_masked(self, ridxs, seed0=0):
        self.step_masked(ridxs[0], seed0)              # warmup
        with wp.ScopedCapture(device=self.dev) as cap:
            for t, r in enumerate(ridxs):
                self.step_masked(r, seed0 + t)
        self.graph = cap.graph

    def replay(self):
        wp.capture_launch(self.graph)
