import argparse
import json
import os
import time
from fractions import Fraction
from common import RESULTS
from tubesparse.instances import zn_instance, primitive_count_zn
from tubesparse.oracle import PromiseUSVPOracle
from tubesparse.reductions import zsvp_deterministic, zsvp_randomized
from tubesparse.isolation import pencil_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="4,5,6,7,8")
    ap.add_argument("--gamma2", default="2,3,4")
    ap.add_argument("--instances", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(RESULTS, "zsvp.jsonl"))
    a = ap.parse_args()
    if os.path.exists(a.out):
        os.remove(a.out)
    for n in [int(x) for x in a.dims.split(",")]:
        for g2 in [int(x) for x in a.gamma2.split(",")]:
            if g2 > n:
                continue
            A = primitive_count_zn(n, g2)
            for inst in range(1, a.instances + 1):
                rows = zn_instance(n, 40 * n, 3, 100 * n + inst)
                oracle = PromiseUSVPOracle(Fraction(g2))
                t0 = time.perf_counter()
                best, calls, p = zsvp_deterministic(rows, oracle, g2, A)
                t_det = time.perf_counter() - t0
                oracle2 = PromiseUSVPOracle(Fraction(g2))
                t0 = time.perf_counter()
                best2, calls2, p2 = zsvp_randomized(rows, oracle2, g2, A, 17 * inst + n)
                t_rnd = time.perf_counter() - t0
                rec = {"n": n, "gamma2": g2, "A": A, "inst": inst, "prime": p, "family_size": pencil_size(p), "det_calls": calls, "det_norm": best[0] if best else None, "det_seconds": t_det, "rnd_calls": calls2, "rnd_norm": best2[0] if best2 else None, "rnd_seconds": t_rnd, "max_bits": oracle.max_bits}
                with open(a.out, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(rec, flush=True)


if __name__ == "__main__":
    main()
