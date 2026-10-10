import argparse
import os
import time
from fractions import Fraction
from multiprocessing import Pool
from common import RESULTS, append_jsonl, read_jsonl
from tubesparse.instances import zn_instance, primitive_count_zn
from tubesparse.oracle import PromiseUSVPOracle
from tubesparse.reductions import zsvp_bgps


def job(args):
    n, g2, A, inst = args
    rows = zn_instance(n, 40 * n, 3, 100 * n + inst)
    oracle = PromiseUSVPOracle(Fraction(g2))
    t0 = time.perf_counter()
    best, calls, p = zsvp_bgps(rows, oracle, A, n, 7777 * inst + n)
    return {"n": n, "gamma2": g2, "A": A, "inst": inst, "prime": p, "calls": calls, "norm": best[0] if best else None, "seconds": time.perf_counter() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="4,5,6,7,8,9,10,11,12")
    ap.add_argument("--gamma2", default="2,3,4")
    ap.add_argument("--instances", type=int, default=5)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=os.path.join(RESULTS, "bgps.jsonl"))
    a = ap.parse_args()
    done = {(r["n"], r["gamma2"], r["inst"]) for r in read_jsonl(a.out)}
    tasks = []
    for n in [int(x) for x in a.dims.split(",")]:
        for g2 in [int(x) for x in a.gamma2.split(",")]:
            if g2 > n or (n >= 11 and g2 >= 4):
                continue
            A = primitive_count_zn(n, g2)
            for inst in range(1, a.instances + 1):
                if (n, g2, inst) not in done:
                    tasks.append((n, g2, A, inst))
    tasks.sort(key=lambda t: -t[2])
    t0 = time.time()
    with Pool(a.workers) as pool:
        for res in pool.imap_unordered(job, tasks):
            append_jsonl(a.out, res)
            print(res["n"], res["gamma2"], res["inst"], res["prime"], res["calls"], round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
