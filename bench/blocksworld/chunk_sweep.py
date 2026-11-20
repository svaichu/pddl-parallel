
import sys, time, numpy as np, jax
from ground import ground, make_pool
from env_jax_tuned import JaxTuned
N=int(sys.argv[1]); E=int(sys.argv[2]); T=int(sys.argv[3]); Cs=[int(x) for x in sys.argv[4:]]
pre,add,dele=ground(N); ps,pg=make_pool(N); A,W=pre.shape
rng=np.random.default_rng(1); rid=rng.integers(0,len(ps),(T,E)).astype(np.int32)
jr=jax.device_put(rid)
env=JaxTuned(pre,add,dele,ps,pg,"aos")
for C in Cs:
    assert E%C==0
    st=[env.to_layout(ps[rid[0]]),env.to_layout(pg[rid[0]])]
    def f():
        out=env.look(st[0],st[1],jr,np.uint32(123),C); st[0],st[1]=out[0],out[1]; jax.block_until_ready(out)
    t0=time.perf_counter(); f(); comp=time.perf_counter()-t0
    ts=[]
    for _ in range(5):
        t0=time.perf_counter(); f(); ts.append(time.perf_counter()-t0)
    m=sorted(ts)[2]; print(f"N={N} A={A} W={W} E={E} C={C:>6} tensor={C*A*W*8/2**20:8.1f}MiB compile={comp:5.1f}s  {E*T/m/1e6:8.2f} Msteps/s  {E*T*A/m/1e9:6.2f}G children/s",flush=True)
