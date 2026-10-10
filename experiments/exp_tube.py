import argparse
import json
import os
import random
from math import sqrt
from common import RESULTS
from tubesparse.instances import random_lattice
from tubesparse.lattice import shortest_vector
from tubesparse.tube import tube_points, tube_bound, ball_union_bound


def directions(kind, n, s, lam, seed):
    if kind == "adv":
        return [x / lam for x in s]
    if kind == "diag":
        return [1.0 / sqrt(n)] * n
    rng = random.Random(seed)
    g = [rng.gauss(0.0, 1.0) for _ in range(n)]
    ng = sqrt(sum(x * x for x in g))
    return [x / ng for x in g]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="10,15,20,25,30,35,40")
    ap.add_argument("--eps", default="0.05,0.1,0.2")
    ap.add_argument("--instances", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(RESULTS, "tube.jsonl"))
    a = ap.parse_args()
    if os.path.exists(a.out):
        os.remove(a.out)
    for n in [int(x) for x in a.dims.split(",")]:
        for family in ["qary", "zn"]:
            for inst in range(1, a.instances + 1):
                if family == "qary":
                    rows = random_lattice(n, 12, 900 * n + inst)
                else:
                    rows = [[int(i == j) for j in range(n)] for i in range(n)]
                lam2, s = shortest_vector(rows)
                lam = sqrt(lam2)
                for eps in [float(x) for x in a.eps.split(",")]:
                    r = lam * sqrt((1.0 - 0.895 * eps) / 2.0)
                    hl = r / sqrt(eps)
                    for kind in ["adv", "rand", "diag"]:
                        if family == "qary" and kind == "diag":
                            continue
                        dr = directions(kind, n, s, lam, 31 * n + inst)
                        pts = tube_points(rows, dr, hl, r, 0.05 * lam)
                        rec = {"n": n, "family": family, "inst": inst, "eps": eps, "dir": kind, "count": len(pts), "bound": tube_bound(n, hl, r, lam), "ball_g1": ball_union_bound(n, 1.0), "ball_g2": ball_union_bound(n, 2.0)}
                        with open(a.out, "a") as fh:
                            fh.write(json.dumps(rec) + "\n")
                        print(rec, flush=True)
                if family == "zn":
                    break


if __name__ == "__main__":
    main()
