"""Slow reference step with python-int bitsets."""
import numpy as np

def _toint(row):
    return sum(int(w) << (64 * i) for i, w in enumerate(row))

def _tonp(v, W):
    return np.array([(v >> (64 * i)) & ((1 << 64) - 1) for i in range(W)], np.uint64)

def step(state, goal, action, ridx, pre, add, dele, pool_s, pool_g):
    E, W = state.shape
    ns, ng = state.copy(), goal.copy()
    rew, done = np.zeros(E, np.float32), np.zeros(E, bool)
    for e in range(E):
        s, g, a = _toint(state[e]), _toint(goal[e]), action[e]
        p, ad, d = _toint(pre[a]), _toint(add[a]), _toint(dele[a])
        if s & p == p:
            s = (s & ~d) | ad
        done[e] = s & g == g
        rew[e] = float(done[e])
        if done[e]:
            ns[e], ng[e] = pool_s[ridx[e]], pool_g[ridx[e]]
        else:
            ns[e] = _tonp(s, W)
    return ns, ng, rew, done
