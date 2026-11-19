"""Merge results_N*.json into results.md. Usage: python report.py results"""
import glob, json, sys, os
d = sys.argv[1] if len(sys.argv) > 1 else "results"
runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "results_N*.json")), key=lambda f: int(f.split("_N")[1][:-5]))]
out = [f"# Blocksworld JAX vs Warp (fair benchmark)\n"]
if runs:
    i = runs[0]["info"]; out.append(f"JAX {i['jax']}, warp-lang {i['warp']}, GPU: {i.get('nvidia_smi') or runs[0]['device']}, git {i.get('git')}, T={runs[0]['T']}\n")
for r in runs:
    keys = sorted({k for row in r["rows"] for k, v in row.items() if isinstance(v, dict)})
    for prefix, title in (("fixed_", "fixed random actions"), ("look_", "greedy 1-step lookahead")):
        ks = [k for k in keys if k.startswith(prefix)]
        out.append(f"\n## N={r['N']} (W={r['W']}, A={r['A']}) - {title}, env-steps/s ({r['device']})\n")
        out.append("| E | " + " | ".join(k[len(prefix):] for k in ks) + " | done rate |\n|" + "---|" * (len(ks) + 2))
        for row in r["rows"]:
            cells = [f"{row[k]['steps_per_s']:.3g}" if k in row else "n/a" for k in ks]
            dr = row.get("done_rate_" + prefix[:-1], float("nan"))
            out.append(f"| {row['E']} | " + " | ".join(cells) + f" | {dr:.4f} |")
open(os.path.join(d, "results.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
