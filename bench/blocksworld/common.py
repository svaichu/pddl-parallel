"""Shared helpers: deterministic tie-break hash (identical in numpy/JAX/Warp), GPU assertion, env info."""
import subprocess

# murmur3-style hash constants (uint32 wrap-around arithmetic everywhere)
C1, C2, C3, C4 = 2654435761, 2246822507, 3266489917, 668265263
F1, F2 = 2246822507, 3266489917
M32 = 0xFFFFFFFF

def hash_py(seed, t, e, a):
    """Reference scalar implementation (python ints)."""
    x = ((seed * C1) & M32) ^ ((t * C2) & M32) ^ ((e * C3) & M32) ^ ((a * C4) & M32)
    x ^= x >> 16; x = (x * F1) & M32; x ^= x >> 13; x = (x * F2) & M32; x ^= x >> 16
    return x

def require_same_gpu():
    """Fail loudly unless JAX and Warp both run on a CUDA GPU (guards against silent JAX-CPU fallback)."""
    import jax, warp as wp
    wp.init()
    jd = jax.devices()[0]
    assert jd.platform in ("gpu", "cuda"), f"JAX is on {jd.platform} ({jd}); need jax[cuda12] and a GPU node"
    assert wp.is_cuda_available(), "Warp sees no CUDA device"
    print(f"JAX device: {jd} ({getattr(jd, 'device_kind', '?')}); Warp device: {wp.get_device('cuda:0').name}", flush=True)

def env_info():
    import jax, warp as wp
    info = {"jax": jax.__version__, "warp": wp.config.version,
            "jax_devices": [str(d) for d in jax.devices()]}
    def sh(cmd):
        try: return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception as ex: return f"n/a ({ex})"
    info["nvidia_smi"] = sh("nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,memory.total --format=csv,noheader")
    info["git"] = sh("git rev-parse --short HEAD")
    info["slurm_job"] = sh("echo $SLURM_JOB_ID")
    return info
