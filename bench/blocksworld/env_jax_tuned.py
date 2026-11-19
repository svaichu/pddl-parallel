"""Tuned JAX env. Differences from env_jax.JaxEnv (the naive baseline):
  - buffer donation (state/goal updated in place), no stacked [T,E] outputs (final reward + done counter only)
  - lax.fori_loop with actions indexed on device (no eager slicing)
  - optional struct-of-arrays layout [W,E]
  - greedy one-step lookahead workload, chunked over envs to bound the [C,A,W] successor tensor
"""
import numpy as np, jax, jax.numpy as jnp
from jax import lax
from common import C1, C2, C3, C4, F1, F2
jax.config.update("jax_enable_x64", True)
u32 = jnp.uint32

def hash32(seed, t, e, a):
    x = (u32(seed) * u32(C1)) ^ (u32(t) * u32(C2)) ^ (e.astype(u32) * u32(C3)) ^ (a.astype(u32) * u32(C4))
    x = x ^ (x >> u32(16)); x = x * u32(F1); x = x ^ (x >> u32(13)); x = x * u32(F2); x = x ^ (x >> u32(16))
    return x

class JaxTuned:
    def __init__(self, pre, add, dele, pool_s, pool_g, layout="aos", chunk_bytes=2**28):
        assert layout in ("aos", "soa")
        self.soa, self.layout, self.chunk_bytes = layout == "soa", layout, chunk_bytes
        self.A, self.W = pre.shape
        j = jnp.asarray
        T_ = lambda x: j(np.ascontiguousarray(x.T))
        self.pre, self.add, self.dele, self.ps, self.pg = map(j, (pre, add, dele, pool_s, pool_g))
        self.preT, self.addT, self.delT, self.psT, self.pgT = map(T_, (pre, add, dele, pool_s, pool_g))
        self.rollout = jax.jit(self._rollout, donate_argnums=(0, 1))
        self.look = jax.jit(self._look, donate_argnums=(0, 1), static_argnums=(4,))

    # layout helpers (numpy [E,W] <-> device layout)
    def to_layout(self, x):
        return jnp.asarray(np.ascontiguousarray(x.T) if self.soa else x)
    def to_aos(self, x):
        x = np.asarray(x)
        return np.ascontiguousarray(x.T) if self.soa else x
    def chunk_for(self, E):
        cap = max(1, self.chunk_bytes // (self.A * self.W * 8))
        c = min(E, cap)
        while E % c: c -= 1
        return c

    def _step_aos(self, s, g, a, r):
        p = self.pre[a]
        app = ((s & p) == p).all(-1)
        ns = jnp.where(app[:, None], (s & ~self.dele[a]) | self.add[a], s)
        done = ((ns & g) == g).all(-1)
        d = done[:, None]
        return jnp.where(d, self.ps[r], ns), jnp.where(d, self.pg[r], g), done

    def _step_soa(self, s, g, a, r):                   # s, g: [W, E]
        p = self.preT[:, a]
        app = ((s & p) == p).all(0)
        ns = jnp.where(app[None], (s & ~self.delT[:, a]) | self.addT[:, a], s)
        done = ((ns & g) == g).all(0)
        d = done[None]
        return jnp.where(d, self.psT[:, r], ns), jnp.where(d, self.pgT[:, r], g), done

    def _step(self, s, g, a, r):
        return (self._step_soa if self.soa else self._step_aos)(s, g, a, r)

    def _rollout(self, s, g, actions, ridxs):          # actions/ridxs [T,E] -> (s, g, ndone[E], last reward[E])
        T, E = actions.shape
        def body(t, c):
            s, g, nd, _ = c
            ns, ng, d = self._step(s, g, actions[t], ridxs[t])
            return ns, ng, nd + d.astype(jnp.int32), d.astype(jnp.float32)
        return lax.fori_loop(0, T, body, (s, g, jnp.zeros(E, jnp.int32), jnp.zeros(E, jnp.float32)))

    def _look(self, s, g, ridxs, seed, C):             # greedy 1-step lookahead, T steps
        T, E = ridxs.shape
        nC = E // C
        if self.soa: s, g = s.T, g.T
        eidx = jnp.arange(E, dtype=jnp.int32).reshape(nC, C)
        arA = jnp.arange(self.A, dtype=jnp.int32)
        def body(t, c):
            s, g, nd, _ = c
            def chunk(x):
                sc, gc, ec, rc = x
                app = ((sc[:, None, :] & self.pre[None]) == self.pre[None]).all(-1)          # [C,A]
                child = (sc[:, None, :] & ~self.dele[None]) | self.add[None]                  # [C,A,W]
                h = lax.population_count(gc[:, None, :] & ~child).sum(-1).astype(jnp.uint32)
                noise = hash32(seed, t, ec[:, None], arA[None]) >> u32(12)
                key = jnp.where(app, (h << u32(20)) | noise, u32(0xFFFFFFFF))
                return self._step_aos(sc, gc, jnp.argmin(key, -1).astype(jnp.int32), rc)
            ns, ng, d = lax.map(chunk, (s.reshape(nC, C, -1), g.reshape(nC, C, -1), eidx, ridxs[t].reshape(nC, C)))
            d = d.reshape(E)
            return ns.reshape(E, -1), ng.reshape(E, -1), nd + d.astype(jnp.int32), d.astype(jnp.float32)
        s, g, nd, rew = lax.fori_loop(0, T, body, (s, g, jnp.zeros(E, jnp.int32), jnp.zeros(E, jnp.float32)))
        if self.soa: s, g = s.T, g.T
        return s, g, nd, rew
