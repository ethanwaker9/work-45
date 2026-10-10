import argparse
import os
import time
from multiprocessing import Pool
from fractions import Fraction
from common import RESULTS, GAMMAS, EPS, regions, gamma_value, append_jsonl, read_jsonl
from tubesparse.instances import random_lattice, bdd_instance
from tubesparse.oracle import PromiseUSVPOracle
from tubesparse.reductions import run_method

METHODS = ["KAN", "LM09", "FOLK", "TSE", "BSW16"]


def job(args):
    n, gkey, region, alpha, mode, inst_id, method, budget = args
    gamma = gamma_value(gkey)
    rows = random_lattice(n, 10, 1000 * n + inst_id)
    inst = bdd_instance(rows, alpha, mode, 7919 * inst_id + n)
    oracle = PromiseUSVPOracle(GAMMAS[gkey])
    res = run_method(method, inst, oracle, gamma, 104729 * inst_id + 31 * n, budget_factor=budget, eps=EPS)
    res.update({"n": n, "gamma": gkey, "region": region, "alpha": alpha, "mode": mode, "inst": inst_id, "d2": inst["d2"], "lam2": inst["lam2"]})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="12,16,20,24,28")
    ap.add_argument("--gammas", default="1.25,1.5,2,3")
    ap.add_argument("--instances", type=int, default=5)
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=os.path.join(RESULTS, "bdd.jsonl"))
    a = ap.parse_args()
    done = {(r["n"], r["gamma"], r["region"], r["mode"], r["inst"], r["method"]) for r in read_jsonl(a.out)}
    tasks = []
    for n in [int(x) for x in a.dims.split(",")]:
        for gkey in a.gammas.split(","):
            for region, alpha in regions(gamma_value(gkey)):
                for mode in ["adv", "rand"]:
                    for inst_id in range(1, a.instances + 1):
                        for method in METHODS:
                            key = (n, gkey, region, mode, inst_id, method)
                            if key not in done:
                                tasks.append((n, gkey, region, alpha, mode, inst_id, method, a.budget))
    tasks.sort(key=lambda t: (t[6] == "BSW16", t[0]))
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(job, tasks)):
            append_jsonl(a.out, res)
            if (i + 1) % 50 == 0:
                print(i + 1, len(tasks), round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
