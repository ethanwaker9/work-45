import argparse
import os
import time
from math import sqrt
from multiprocessing import Pool
from common import RESULTS, GAMMAS, gamma_value, append_jsonl, read_jsonl
from tubesparse.instances import random_lattice, bdd_instance
from tubesparse.oracle import PromiseUSVPOracle
from tubesparse.reductions import single_call

GA = [0.40 + 0.01 * i for i in range(25)]
KS = [10 ** (-1.5 + 2.2 * i / 44) for i in range(45)]


def job(args):
    n, gkey, inst_id, ga = args
    gamma = gamma_value(gkey)
    rows = random_lattice(n, 10, 1000 * n + inst_id)
    inst = bdd_instance(rows, ga / gamma, "adv", 7919 * inst_id + n)
    out = []
    for k in KS:
        oracle = PromiseUSVPOracle(GAMMAS[gkey])
        out.append(single_call(inst, oracle, k))
    return {"n": n, "gamma": gkey, "inst": inst_id, "ga": ga, "ks": KS, "ok": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=os.path.join(RESULTS, "fold.jsonl"))
    a = ap.parse_args()
    done = {(r["n"], r["gamma"], r["inst"], round(r["ga"], 4)) for r in read_jsonl(a.out)}
    tasks = [(20, g, i, ga) for g in ["1.5", "2", "3"] for i in range(1, 4) for ga in GA if (20, g, i, round(ga, 4)) not in done]
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(job, tasks)):
            append_jsonl(a.out, res)
    print("fold", len(tasks), round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
