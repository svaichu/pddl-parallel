"""Blocksworld grounder + problem pool. Facts: on(x,y) x!=y, ontable, clear, holding, handempty -> (N+1)^2."""
import numpy as np

def layout(N):
    on = {}
    k = 0
    for x in range(N):
        for y in range(N):
            if x != y:
                on[(x, y)] = k; k += 1
    ot, cl, ho = N*(N-1), N*(N-1)+N, N*(N-1)+2*N
    return on, ot, cl, ho, N*(N-1)+3*N   # last = handempty; total = (N+1)^2

def _mask(bits, W):
    m = np.zeros(W, np.uint64)
    for b in bits:
        m[b // 64] |= np.uint64(1) << np.uint64(b % 64)
    return m

def ground(N):
    """Returns pre, add, del: uint64[A, W]; A = 2N^2."""
    on, ot, cl, ho, he = layout(N)
    F = he + 1; W = (F + 63) // 64
    acts = []
    for x in range(N):
        acts.append(([ot+x, cl+x, he], [ho+x], [ot+x, cl+x, he]))                 # pickup
    for x in range(N):
        acts.append(([ho+x], [ot+x, cl+x, he], [ho+x]))                           # putdown
    for x in range(N):
        for y in range(N):
            if x != y:
                acts.append(([ho+x, cl+y], [on[x, y], cl+x, he], [ho+x, cl+y]))   # stack
    for x in range(N):
        for y in range(N):
            if x != y:
                acts.append(([on[x, y], cl+x, he], [ho+x, cl+y], [on[x, y], cl+x, he]))  # unstack
    pre, add, dele = (np.stack([_mask(a[i], W) for a in acts]) for i in range(3))
    return pre, add, dele

def _random_state(N, rng, W):
    """Random towers, hand empty. Returns (full state bits, on-fact bits)."""
    on, ot, cl, ho, he = layout(N)
    perm = rng.permutation(N)
    cuts = rng.random(N - 1) < 0.5          # tower boundaries
    bits, onb = [he], []
    for i, b in enumerate(perm):
        if i == 0 or cuts[i - 1]:
            bits.append(ot + b)
        else:
            bits.append(on[b, perm[i - 1]]); onb.append(on[b, perm[i - 1]])
        if i == N - 1 or cuts[i]:
            bits.append(cl + b)
    return _mask(bits, W), _mask(onb, W)

def make_pool(N, K=4096, seed=0):
    """Returns pool_state[K,W], pool_goal[K,W] (goal = on-facts of a random state)."""
    W = ground(N)[0].shape[1]
    rng = np.random.default_rng(seed)
    ps, pg = [], []
    for _ in range(K):
        ps.append(_random_state(N, rng, W)[0]); pg.append(_random_state(N, rng, W)[1])
    return np.stack(ps), np.stack(pg)
