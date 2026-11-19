import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

class JaxEnv:
    def __init__(self, pre, add, dele, pool_s, pool_g):
        self.pre, self.add, self.dele = map(jnp.asarray, (pre, add, dele))
        self.ps, self.pg = jnp.asarray(pool_s), jnp.asarray(pool_g)
        self.K = self.ps.shape[0]
        self.mask = jax.jit(self._mask)
        self.step = jax.jit(self._step)
        self.rollout = jax.jit(self._rollout)
        self.rollout_masked = jax.jit(self._rollout_masked)

    def reset_all(self, seed, E):
        idx = jax.random.randint(jax.random.PRNGKey(seed), (E,), 0, self.K)
        return self.ps[idx], self.pg[idx]

    def _mask(self, s):
        return ((s[:, None, :] & self.pre[None]) == self.pre[None]).all(-1)

    def _step(self, s, g, a, ridx):
        p = self.pre[a]
        app = ((s & p) == p).all(-1)
        ns = jnp.where(app[:, None], (s & ~self.dele[a]) | self.add[a], s)
        done = ((ns & g) == g).all(-1)
        d = done[:, None]
        return (jnp.where(d, self.ps[ridx], ns), jnp.where(d, self.pg[ridx], g),
                done.astype(jnp.float32), done)

    def _rollout(self, s, g, actions, ridxs):          # actions/ridxs: [T, E]
        def body(c, x):
            ns, ng, r, d = self._step(*c, *x)
            return (ns, ng), (r, d)
        return jax.lax.scan(body, (s, g), (actions, ridxs))

    def _rollout_masked(self, s, g, key, ridxs):       # masked uniform policy; ridxs: [T, E]
        def body(c, x):
            s, g = c
            k, r = x
            logits = jnp.where(self._mask(s), 0.0, -jnp.inf)
            a = jax.random.categorical(k, logits, axis=-1)
            ns, ng, rew, d = self._step(s, g, a, r)
            return (ns, ng), (rew, d)
        keys = jax.random.split(key, ridxs.shape[0])
        return jax.lax.scan(body, (s, g), (keys, ridxs))
