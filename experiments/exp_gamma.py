import argparse
import os
import time
from fractions import Fraction
from math import sqrt
from multiprocessing import Pool
from common import RESULTS, EPS, append_jsonl, read_jsonl
from tubesparse.instances import random_lattice, bdd_instance
from tubesparse.oracle import PromiseUSVPOracle
from tubesparse.reductions import run_method, folk_alpha, sqrt2_alpha

GAMMA2 = [Fraction(1), Fraction(4), Fraction(16), Fraction(64), Fraction(256), Fraction(1024), Fraction(4096)]


def job(args):
    n, g2, inst_id, method, budget = args
    gamma = sqrt(float(g2))
    a_f = folk_alpha(gamma)
    a_t = (1.0 - EPS) * sqrt2_alpha(gamma)
    alpha = 0.5 * (a_f + a_t)
    rows = random_lattice(n, 10, 5000 * n + inst_id)
    inst = bdd_instance(rows, alpha, "adv", 6007 * inst_id + n)
    oracle = PromiseUSVPOracle(g2)
    res = run_method(method, inst, oracle, gamma, 7727 * inst_id + n, budget_factor=budget, eps=EPS)
    res.update({"n": n, "gamma2": str(g2), "gamma": gamma, "alpha": alpha, "inst": inst_id})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="12,16")
    ap.add_argument("--instances", type=int, default=10)
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=os.path.join(RESULTS, "gamma.jsonl"))
    a = ap.parse_args()
    done = {(r["n"], r["gamma2"], r["inst"], r["method"]) for r in read_jsonl(a.out)}
    tasks = []
    for n in [int(x) for x in a.dims.split(",")]:
        for g2 in GAMMA2:
            for inst_id in range(1, a.instances + 1):
                for method in ["FOLK", "TSE", "BSW16"]:
                    if (n, str(g2), inst_id, method) not in done:
                        tasks.append((n, g2, inst_id, method, a.budget))
    tasks.sort(key=lambda t: (t[3] == "BSW16", float(t[1])))
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(job, tasks)):
            append_jsonl(a.out, res)
            print(i + 1, len(tasks), res["n"], res["gamma2"], res["method"], res["calls"], res["success"], round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
