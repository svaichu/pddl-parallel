
import sys, time, numpy as np, jax, jax.numpy as jnp
from jax import lax
from ground import ground, make_pool
from env_jax_tuned import JaxTuned, hash32, u32
N=int(sys.argv[1]); E=int(sys.argv[2]); C=int(sys.argv[3])
pre,add,dele=ground(N); ps,pg=make_pool(N); A,W=pre.shape
rng=np.random.default_rng(1); rid=rng.integers(0,len(ps),E)
s=jnp.asarray(ps[rid]); g=jnp.asarray(pg[rid])
P,Ad,D=map(jnp.asarray,(pre,add,dele))
arA=jnp.arange(A,dtype=jnp.int32)
def score_ref(sc,gc):
    app=((sc[:,None,:]&P[None])==P[None]).all(-1)
    child=(sc[:,None,:]&~D[None])|Ad[None]
    return app, lax.population_count(gc[:,None,:]&~child).sum(-1).astype(jnp.uint32)
def score_unroll(sc,gc):
    app=jnp.ones((sc.shape[0],A),bool); h=jnp.zeros((sc.shape[0],A),jnp.uint32)
    for w in range(W):
        sw=sc[:,w,None]; gw=gc[:,w,None]
        app&=((sw&P[:,w][None])==P[:,w][None])
        child=(sw&~D[:,w][None])|Ad[:,w][None]
        h+=lax.population_count(gw&~child).astype(jnp.uint32)
    return app,h
def mk(score):
    def body(x):
        sc,gc,ec=x
        app,h=score(sc,gc)
        noise=hash32(123,0,ec[:,None],arA[None])>>u32(12)
        key=jnp.where(app,(h<<u32(20))|noise,u32(0xFFFFFFFF))
        return jnp.argmin(key,-1).astype(jnp.int32)
    @jax.jit
    def f(s,g):
        nC=E//C
        return lax.map(body,(s.reshape(nC,C,-1),g.reshape(nC,C,-1),jnp.arange(E,dtype=jnp.int32).reshape(nC,C)))
    return f
def score_envminor(sc,gc):                       # sc,gc [C,W] -> compute in [A,W,C], envs minor
    sT=sc.T[None]; gT=gc.T[None]
    Pp,Aa,Dd=P[:,:,None],Ad[:,:,None],D[:,:,None]
    app=((sT&Pp)==Pp).all(1)                      # [A,C]
    child=(sT&~Dd)|Aa
    h=lax.population_count(gT&~child).sum(1).astype(jnp.uint32)
    return app.T,h.T
outs={}
for name,sc in (("ref",score_ref),("envminor",score_envminor)):
    f=mk(sc); o=f(s,g); jax.block_until_ready(o); outs[name]=np.asarray(o)
    ts=[]
    for _ in range(7):
        t0=time.perf_counter(); jax.block_until_ready(f(s,g)); ts.append(time.perf_counter()-t0)
    m=sorted(ts)[3]; print(f"{name:7s} N={N} E={E} C={C}: {m*1e3:7.2f} ms/step  {E*A/m/1e9:6.2f}G children/s",flush=True)
print("identical argmin:", np.array_equal(outs["ref"],outs["envminor"]))
