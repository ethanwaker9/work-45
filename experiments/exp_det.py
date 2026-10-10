import argparse
import os
import time
from multiprocessing import Pool
from common import RESULTS, GAMMAS, EPS, regions, gamma_value, append_jsonl, read_jsonl
from tubesparse.instances import random_lattice, bdd_instance
from tubesparse.oracle import PromiseUSVPOracle
from tubesparse.reductions import bdd_deterministic, det_prime


def job(args):
    n, gkey, region, alpha, inst_id, cap = args
    rows = random_lattice(n, 10, 1000 * n + inst_id)
    inst = bdd_instance(rows, alpha, "adv", 7919 * inst_id + n)
    oracle = PromiseUSVPOracle(GAMMAS[gkey])
    res = bdd_deterministic(inst, oracle, eps=EPS, max_calls=cap * det_prime(n, EPS)[0])
    res.update({"n": n, "gamma": gkey, "region": region, "alpha": alpha, "mode": "adv", "inst": inst_id, "d2": inst["d2"], "lam2": inst["lam2"]})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="12,16,20")
    ap.add_argument("--gammas", default="1.25,1.5,2,3")
    ap.add_argument("--instances", type=int, default=5)
    ap.add_argument("--cap", type=int, default=5)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=os.path.join(RESULTS, "det.jsonl"))
    a = ap.parse_args()
    done = {(r["n"], r["gamma"], r["region"], r["inst"]) for r in read_jsonl(a.out)}
    tasks = []
    for n in [int(x) for x in a.dims.split(",")]:
        for gkey in a.gammas.split(","):
            for region, alpha in regions(gamma_value(gkey)):
                if not region.startswith("R3"):
                    continue
                for inst_id in range(1, a.instances + 1):
                    if (n, gkey, region, inst_id) not in done:
                        tasks.append((n, gkey, region, alpha, inst_id, a.cap))
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(job, tasks)):
            append_jsonl(a.out, res)
            print(i + 1, len(tasks), res["n"], res["gamma"], res["region"], res["calls"], res["success"], round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
