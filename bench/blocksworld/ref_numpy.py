"""Slow reference step with python-int bitsets."""
import numpy as np
from common import hash_py

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

def lookahead_actions(state, goal, pre, add, dele, seed, t):
    """Greedy one-step lookahead: argmin over applicable a of key = (popcount(goal & ~child) << 20) | (hash >> 12)."""
    E, W = state.shape
    A = pre.shape[0]
    P, AD, D = [_toint(x) for x in pre], [_toint(x) for x in add], [_toint(x) for x in dele]
    acts = np.zeros(E, np.int32)
    for e in range(E):
        s, g = _toint(state[e]), _toint(goal[e])
        best, ba = 0xFFFFFFFF, 0
        for a in range(A):
            if s & P[a] == P[a]:
                ch = (s & ~D[a]) | AD[a]
                h = bin(g & ~ch).count("1")
                key = (h << 20) | (hash_py(seed, t, e, a) >> 12)
                if key < best:
                    best, ba = key, a
        acts[e] = ba
    return acts
