import argparse
import os
import random
import time
from fractions import Fraction
from math import sqrt, ceil
from multiprocessing import Pool
from common import RESULTS, GAMMAS, EPS, regions, gamma_value, append_jsonl
from tubesparse.instances import random_lattice, bdd_instance
from tubesparse.lattice import shortest_list, lll_rows, parallel, norm2, rational_inverse, coords, combine, add
from tubesparse.reductions import embedding, folk_k
from tubesparse.isolation import tse_prime, sparsified_basis
from tubesparse.tube import unwanted_exact


def gap_of(rows, gamma):
    red = lll_rows(rows)
    b2 = norm2(red[0])
    cap = (gamma + 1.0) ** 2
    sols, _ = shortest_list(red, 2 * int(ceil(gamma + 1)) + 3, radius2=int(cap * b2) + 1)
    n1, v1 = sols[0]
    for nn, v in sols[1:]:
        if not parallel(v, v1):
            return sqrt(nn / n1), v1
    return None, v1


def job(args):
    n, gkey, region, inst_id = args
    gamma = gamma_value(gkey)
    alpha = dict(regions(gamma))[region]
    rows = random_lattice(n, 10, 1000 * n + inst_id)
    inst = bdd_instance(rows, alpha, "adv", 7919 * inst_id + n)
    d = sqrt(inst["d2"])
    out = {"n": n, "gamma": gkey, "region": region, "inst": inst_id}
    for name, k in [("LM09", 1.0), ("FOLK", folk_k(gamma)), ("KAN", 1.0 / sqrt(n))]:
        m = max(1, int(round(k * d)))
        g, v1 = gap_of(embedding(inst["B"], inst["t"], m), gamma)
        out[name] = {"gap": g, "wanted": abs(v1[-1]) == m and norm2(v1[:-1]) == inst["d2"]}
    p = tse_prime(n, EPS)
    m = max(1, int(round(sqrt(EPS) * d)))
    Y = unwanted_exact(inst["B"], inst["e"], m, GAMMAS[gkey])
    inv = rational_inverse(inst["B"])
    cbar = [int(x) % p for x in coords(inst["c"], inv)]
    ybars = [[int(x) % p for x in coords(y, inv)] for y in Y]
    rng = random.Random(97 * inst_id + n)
    while True:
        u = [rng.randrange(p) for _ in range(n)]
        z = [rng.randrange(p) for _ in range(n)]
        if not any(z):
            continue
        x = [(a + b) % p for a, b in zip(cbar, u)]
        if sum(a * b for a, b in zip(z, x)) % p != 0:
            continue
        if all(sum(a * b for a, b in zip(z, yb)) % p != 0 for yb in ybars):
            break
    w = combine(u, inst["B"])
    g, v1 = gap_of(embedding(sparsified_basis(inst["B"], z, p), add(inst["t"], w), m), gamma)
    out["TSE"] = {"gap": g, "wanted": abs(v1[-1]) == m and norm2(v1[:-1]) == inst["d2"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=os.path.join(RESULTS, "uniq.jsonl"))
    a = ap.parse_args()
    if os.path.exists(a.out):
        os.remove(a.out)
    tasks = [(n, g, reg, i) for n in [12, 16, 20, 24, 28] for g in ["1.25", "1.5", "2", "3"] for reg in ["R3a", "R3b"] for i in range(1, 6)]
    t0 = time.time()
    with Pool(a.workers) as pool:
        for res in pool.imap_unordered(job, tasks):
            append_jsonl(a.out, res)
    print("uniq", len(tasks), round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
