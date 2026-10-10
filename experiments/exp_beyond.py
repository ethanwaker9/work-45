import argparse
import os
import time
from math import sqrt
from multiprocessing import Pool
from common import RESULTS, GAMMAS, EPS, gamma_value, append_jsonl, read_jsonl
from tubesparse.instances import random_lattice, bdd_instance
from tubesparse.oracle import PromiseUSVPOracle
from tubesparse.reductions import run_method
from tubesparse.tube import unwanted_exact
from tubesparse.lattice import shortest_vector, parallel

GAS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def job(args):
    n, gkey, ga, inst_id, method = args
    gamma = gamma_value(gkey)
    rows = random_lattice(n, 10, 1000 * n + inst_id)
    inst = bdd_instance(rows, ga / gamma, "adv", 7919 * inst_id + n)
    if method == "Y":
        m = max(1, int(round(sqrt(EPS * inst["d2"]))))
        Y = unwanted_exact(inst["B"], inst["e"], m, GAMMAS[gkey], cap=20000)
        _, sv = shortest_vector(inst["B"])
        return {"n": n, "gamma": gkey, "ga": ga, "inst": inst_id, "method": "Y", "Y": len(Y), "offline": sum(1 for y in Y if not parallel(y, sv))}
    oracle = PromiseUSVPOracle(GAMMAS[gkey])
    res = run_method(method, inst, oracle, gamma, 104729 * inst_id + 31 * n, budget_factor=10, eps=EPS)
    res.update({"n": n, "gamma": gkey, "ga": ga, "inst": inst_id})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=os.path.join(RESULTS, "beyond.jsonl"))
    a = ap.parse_args()
    done = {(r["ga"], r["inst"], r["method"]) for r in read_jsonl(a.out)}
    tasks = [(20, "2", ga, i, m) for ga in GAS for i in range(1, 6) for m in ["Y", "FOLK", "ATSE", "TSE", "BSW16"] if (ga, i, m) not in done]
    tasks.sort(key=lambda t: t[4] == "BSW16")
    t0 = time.time()
    with Pool(a.workers) as pool:
        for res in pool.imap_unordered(job, tasks):
            append_jsonl(a.out, res)
    print("beyond", len(tasks), round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
