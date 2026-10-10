import argparse
import json
import os
import time
from fractions import Fraction
from common import RESULTS
from tubesparse.instances import zn_instance, primitive_count_zn
from tubesparse.oracle import PromiseUSVPOracle
from tubesparse.reductions import zsvp_thin, zsvp_deterministic, zsvp_randomized
from tubesparse.isolation import thin_pencil_size, pencil_size, isolation_prime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="4,5,6,7,8,9,10,11,12")
    ap.add_argument("--gamma2", default="2,3,4")
    ap.add_argument("--instances", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(RESULTS, "thin.jsonl"))
    a = ap.parse_args()
    if os.path.exists(a.out):
        os.remove(a.out)
    for n in [int(x) for x in a.dims.split(",")]:
        for g2 in [int(x) for x in a.gamma2.split(",")]:
            if g2 > n or (n >= 11 and g2 >= 4):
                continue
            A = primitive_count_zn(n, g2)
            for inst in range(1, a.instances + 1):
                rows = zn_instance(n, 40 * n, 3, 100 * n + inst)
                rec = {"n": n, "gamma2": g2, "A": A, "inst": inst}
                for name, fn in [("thin", zsvp_thin), ("pencil", zsvp_deterministic)]:
                    oracle = PromiseUSVPOracle(Fraction(g2))
                    t0 = time.perf_counter()
                    best, calls, p = fn(rows, oracle, g2, A)
                    rec[name] = {"calls": calls, "prime": p, "norm": best[0] if best else None, "seconds": time.perf_counter() - t0, "bits": oracle.max_bits}
                rec["thin"]["family"] = thin_pencil_size(n, rec["thin"]["prime"])
                rec["pencil"]["family"] = pencil_size(rec["pencil"]["prime"])
                oracle = PromiseUSVPOracle(Fraction(g2))
                best, calls, p = zsvp_randomized(rows, oracle, g2, A, 17 * inst + n)
                rec["random"] = {"calls": calls, "prime": p, "norm": best[0] if best else None}
                with open(a.out, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(n, g2, inst, rec["thin"]["calls"], rec["pencil"]["calls"], rec["random"]["calls"], flush=True)


if __name__ == "__main__":
    main()
