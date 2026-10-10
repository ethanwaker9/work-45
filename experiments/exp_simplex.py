import argparse
import json
import os
import random
import time
from fractions import Fraction
from math import sqrt, ceil, exp
from multiprocessing import Pool
from common import RESULTS, append_jsonl
from tubesparse.instances import simplex_rows, simplex_bdd_instance
from tubesparse.oracle import PromiseUSVPOracle
from tubesparse.tube import tube_points, unwanted_exact, residue_lines, avoid_probability
from tubesparse.lattice import rational_inverse
from tubesparse.reductions import sparsified_trial, run_method
from tubesparse.isolation import next_prime, tse_prime

GAMMA2 = Fraction(4)


def slice_job(n):
    b = int(ceil(sqrt(n + 1)))
    rows = simplex_rows(n, b)
    u = [1.0 / sqrt(n)] * n
    rho = sqrt(n * (n - 1))
    pts = tube_points(rows, u, b * sqrt(n), rho * (1 + 1e-9), 0.05 * sqrt(2) * n, cap=4 * n + 8)
    sig = {}
    for v in pts:
        s = round(sum(x for x in v) / sqrt(n), 6)
        sig[s] = sig.get(s, 0) + 1
    return {"part": "slice", "n": n, "count": len(pts), "max_slice": max(sig.values()), "eta": 1.0 / n}


def rate_job(args):
    n, p, trials = args
    eps = 1.0 / (40 * n)
    inst = simplex_bdd_instance(n, GAMMA2, eps)
    m = max(1, int(round(sqrt(eps * inst["d2"]))))
    Y = unwanted_exact(inst["B"], inst["e"], m, GAMMA2)
    inv = rational_inverse(inst["B"])
    zero, lines = residue_lines(Y, inv, p)
    pred = 0.0 if zero else float(avoid_probability(lines, n, p)) / p
    bound = (p - 1) ** n / (p ** n - 1) / p
    oracle = PromiseUSVPOracle(GAMMA2)
    rng = random.Random(7 * n + 1000 * p)
    succ = sum(sparsified_trial(inst, oracle, rng, p, m, 0.0) for _ in range(trials))
    return {"part": "rate", "n": n, "p": p, "trials": trials, "succ": succ, "pred": pred, "bound": bound, "Y": len(Y), "lines": len(lines)}


def run_job(args):
    n, method, seed = args
    eps = 1.0 / (40 * n)
    inst = simplex_bdd_instance(n, GAMMA2, eps)
    oracle = PromiseUSVPOracle(GAMMA2)
    res = run_method(method, inst, oracle, 2.0, seed, budget_factor=10, eps=eps)
    res.update({"part": "run", "n": n, "seed": seed, "lower": 0.75 * exp(1) * n})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--parts", default="slice,rate,run")
    ap.add_argument("--out", default=os.path.join(RESULTS, "simplex.jsonl"))
    a = ap.parse_args()
    parts = a.parts.split(",")
    keep = [r for r in (json.loads(l) for l in open(a.out)) if r["part"] not in parts] if os.path.exists(a.out) else []
    with open(a.out, "w") as fh:
        for r in keep:
            fh.write(json.dumps(r) + "\n")
    t0 = time.time()
    with Pool(a.workers) as pool:
        if "slice" in parts:
            for res in pool.imap_unordered(slice_job, list(range(4, 25, 2))):
                append_jsonl(a.out, res)
            print("slice", round(time.time() - t0, 1), flush=True)
        if "rate" in parts:
            tasks = []
            for n in [6, 8, 10, 12]:
                ps = sorted({next_prime(max(2, int(f * n))) for f in [0.5, 1, 2, 4, 8]})
                for p in ps:
                    pr = (1.0 / p) * (1 - 1.0 / p) ** n
                    tasks.append((n, p, int(min(60000, max(2000, 150 / pr)))))
            for res in pool.imap_unordered(rate_job, tasks):
                append_jsonl(a.out, res)
            print("rate", round(time.time() - t0, 1), flush=True)
        if "run" in parts:
            tasks = [(n, m, s) for n in [6, 8, 10, 12] for m in ["ATSE-S", "TSE-S"] for s in range(1, (21 if m == "ATSE-S" else 6))]
            for res in pool.imap_unordered(run_job, tasks):
                append_jsonl(a.out, res)
            print("run", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
