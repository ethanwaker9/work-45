import argparse
import os
import time
from math import sqrt
from multiprocessing import Pool
from common import RESULTS, GAMMAS, regions, gamma_value, append_jsonl
from tubesparse.instances import random_lattice, bdd_instance
from tubesparse.tube import unwanted_exact
from tubesparse.lattice import parallel, shortest_vector

KS = [0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.5, 0.8, 1.0]


def job(args):
    n, gkey, inst_id, k = args
    gamma = gamma_value(gkey)
    alpha = dict(regions(gamma))["R3b"]
    rows = random_lattice(n, 10, 1000 * n + inst_id)
    inst = bdd_instance(rows, alpha, "adv", 7919 * inst_id + n)
    d = sqrt(inst["d2"])
    m = max(1, int(round(k * d)))
    Y = unwanted_exact(inst["B"], inst["e"], m, GAMMAS[gkey], cap=20000)
    _, sv = shortest_vector(inst["B"])
    on_line = sum(1 for y in Y if parallel(y, sv))
    a = sqrt(inst["d2"] / inst["lam2"])
    kk = m / d
    upper = a * gamma * (1 + kk * kk) / kk
    lower = a * (1 + kk * kk) * sqrt(gamma * gamma - 0.25) / kk
    return {"n": n, "gamma": gkey, "inst": inst_id, "k": k, "kk": kk, "alpha": a, "Y": len(Y), "line": on_line, "upper": upper, "lower": lower}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=os.path.join(RESULTS, "mult.jsonl"))
    a = ap.parse_args()
    if os.path.exists(a.out):
        os.remove(a.out)
    tasks = [(n, g, i, k) for n in [12, 20, 28] for g in ["1.5", "2", "3"] for i in range(1, 4) for k in KS]
    t0 = time.time()
    with Pool(a.workers) as pool:
        for res in pool.imap_unordered(job, tasks):
            append_jsonl(a.out, res)
    print("mult", len(tasks), round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
