import argparse
import os
import time
from multiprocessing import Pool
from common import RESULTS, GAMMAS, EPS, regions, gamma_value, append_jsonl, read_jsonl
from tubesparse.instances import random_lattice, bdd_instance
from tubesparse.oracle import PromiseUSVPOracle
from tubesparse.reductions import bdd_shift_deterministic


def job(args):
    n, gkey, region, alpha, inst_id = args
    rows = random_lattice(n, 10, 1000 * n + inst_id)
    inst = bdd_instance(rows, alpha, "adv", 7919 * inst_id + n)
    oracle = PromiseUSVPOracle(GAMMAS[gkey])
    res = bdd_shift_deterministic(inst, oracle, eps=EPS)
    res.update({"n": n, "gamma": gkey, "region": region, "alpha": alpha, "mode": "adv", "inst": inst_id})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="12,16,20,24,28")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=os.path.join(RESULTS, "dets.jsonl"))
    a = ap.parse_args()
    done = {(r["n"], r["gamma"], r["region"], r["inst"]) for r in read_jsonl(a.out)}
    tasks = []
    for n in [int(x) for x in a.dims.split(",")]:
        for gkey in ["1.25", "1.5", "2", "3"]:
            for region, alpha in regions(gamma_value(gkey)):
                if region.startswith("R3"):
                    for inst_id in range(1, 6):
                        if (n, gkey, region, inst_id) not in done:
                            tasks.append((n, gkey, region, alpha, inst_id))
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(job, tasks)):
            append_jsonl(a.out, res)
            if (i + 1) % 20 == 0:
                print(i + 1, len(tasks), round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
