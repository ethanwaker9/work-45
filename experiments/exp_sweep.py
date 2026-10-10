import argparse
import os
import time
from math import sqrt
from multiprocessing import Pool
from common import RESULTS, GAMMAS, EPS, gamma_value, append_jsonl, read_jsonl
from tubesparse.instances import random_lattice, bdd_instance
from tubesparse.oracle import PromiseUSVPOracle
from tubesparse.reductions import run_method, sqrt2_alpha

METHODS = ["KAN", "LM09", "FOLK", "TSE", "BSW16", "ATSE", "ABSW16"]


def alphas(gamma, step, top):
    count = int(round(top / step))
    return [step * (i + 1) / gamma for i in range(count)]


def job(args):
    n, gkey, alpha, idx, inst_id, method, budget = args
    gamma = gamma_value(gkey)
    rows = random_lattice(n, 10, 3000 * n + inst_id)
    inst = bdd_instance(rows, alpha, "adv", 3571 * inst_id + n)
    oracle = PromiseUSVPOracle(GAMMAS[gkey])
    res = run_method(method, inst, oracle, gamma, 9973 * inst_id + n, budget_factor=budget, eps=EPS)
    res.update({"n": n, "gamma": gkey, "alpha": alpha, "idx": idx, "inst": inst_id})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--gammas", default="1.5,2")
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--top", type=float, default=0.70)
    ap.add_argument("--instances", type=int, default=10)
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=os.path.join(RESULTS, "sweep.jsonl"))
    a = ap.parse_args()
    done = {(r["gamma"], r["idx"], r["inst"], r["method"]) for r in read_jsonl(a.out)}
    tasks = []
    for gkey in a.gammas.split(","):
        for idx, alpha in enumerate(alphas(gamma_value(gkey), a.step, a.top)):
            for inst_id in range(1, a.instances + 1):
                for method in METHODS:
                    if (gkey, idx, inst_id, method) not in done:
                        tasks.append((a.n, gkey, alpha, idx, inst_id, method, a.budget))
    tasks.sort(key=lambda t: t[5] == "BSW16")
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(job, tasks)):
            append_jsonl(a.out, res)
            if (i + 1) % 50 == 0:
                print(i + 1, len(tasks), round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
