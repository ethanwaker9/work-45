import argparse
import os
import time
from fractions import Fraction
from math import sqrt
from multiprocessing import Pool
from common import RESULTS, GAMMAS, EPS, regions, gamma_value, append_jsonl, read_jsonl
from tubesparse.instances import random_lattice, bdd_instance
from tubesparse.oracle import PromiseUSVPOracle
from tubesparse.reductions import run_method, folk_alpha, sqrt2_alpha

METHODS = ["ATSE", "ABSW16"]


def job_bdd(args):
    n, gkey, region, alpha, mode, inst_id, method, budget = args
    gamma = gamma_value(gkey)
    rows = random_lattice(n, 10, 1000 * n + inst_id)
    inst = bdd_instance(rows, alpha, mode, 7919 * inst_id + n)
    oracle = PromiseUSVPOracle(GAMMAS[gkey])
    res = run_method(method, inst, oracle, gamma, 104729 * inst_id + 31 * n, budget_factor=budget, eps=EPS)
    res.update({"n": n, "gamma": gkey, "region": region, "alpha": alpha, "mode": mode, "inst": inst_id, "d2": inst["d2"], "lam2": inst["lam2"], "exp": "bdd"})
    return res


def job_gap(args):
    n, g2, inst_id, method, budget = args
    gamma = sqrt(float(g2))
    alpha = 0.5 * (folk_alpha(gamma) + (1.0 - EPS) * sqrt2_alpha(gamma))
    rows = random_lattice(n, 10, 5000 * n + inst_id)
    inst = bdd_instance(rows, alpha, "adv", 6007 * inst_id + n)
    oracle = PromiseUSVPOracle(g2)
    res = run_method(method, inst, oracle, gamma, 7727 * inst_id + n, budget_factor=budget, eps=EPS)
    res.update({"n": n, "gamma2": str(g2), "gamma": gamma, "alpha": alpha, "inst": inst_id, "exp": "gap"})
    return res


def job_large(args):
    n, gkey, inst_id, method, budget = args
    gamma = gamma_value(gkey)
    alpha = dict(regions(gamma))["R3b"]
    rows = random_lattice(n, 10, 1000 * n + inst_id)
    inst = bdd_instance(rows, alpha, "adv", 7919 * inst_id + n)
    oracle = PromiseUSVPOracle(GAMMAS[gkey])
    res = run_method(method, inst, oracle, gamma, 104729 * inst_id + 31 * n, budget_factor=budget, eps=EPS)
    res.update({"n": n, "gamma": gkey, "region": "R3b", "alpha": alpha, "mode": "adv", "inst": inst_id, "exp": "large"})
    return res


def run(pool, fn, tasks, out):
    t0 = time.time()
    for i, res in enumerate(pool.imap_unordered(fn, tasks)):
        append_jsonl(out, res)
        if (i + 1) % 50 == 0 or len(tasks) < 100:
            print(res.get("exp"), i + 1, len(tasks), res["method"], res["n"], res["calls"], res["success"], round(time.time() - t0, 1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--parts", default="bdd,gap,large")
    ap.add_argument("--out", default=os.path.join(RESULTS, "atse.jsonl"))
    a = ap.parse_args()
    rs = read_jsonl(a.out)
    parts = a.parts.split(",")
    with Pool(a.workers) as pool:
        if "bdd" in parts:
            done = {(r["n"], r["gamma"], r["region"], r["mode"], r["inst"], r["method"]) for r in rs if r["exp"] == "bdd"}
            tasks = []
            for n in [12, 16, 20, 24, 28]:
                for gkey in ["1.25", "1.5", "2", "3"]:
                    for region, alpha in regions(gamma_value(gkey)):
                        for mode in ["adv", "rand"]:
                            for inst_id in range(1, 6):
                                for method in METHODS:
                                    if (n, gkey, region, mode, inst_id, method) not in done:
                                        tasks.append((n, gkey, region, alpha, mode, inst_id, method, a.budget))
            run(pool, job_bdd, tasks, a.out)
        if "gap" in parts:
            done = {(r["n"], r["gamma2"], r["inst"], r["method"]) for r in rs if r["exp"] == "gap"}
            tasks = []
            for n in [12, 16]:
                for g2 in [Fraction(4 ** k) for k in range(7)]:
                    for inst_id in range(1, 11):
                        for method in METHODS:
                            if (n, str(g2), inst_id, method) not in done:
                                tasks.append((n, g2, inst_id, method, a.budget))
            run(pool, job_gap, tasks, a.out)
        if "large" in parts:
            done = {(r["n"], r["gamma"], r["inst"], r["method"]) for r in rs if r["exp"] == "large"}
            tasks = []
            for n, methods in [(32, ["FOLK", "ATSE", "ABSW16", "TSE", "BSW16", "TSEF"]), (36, ["FOLK", "ATSE", "ABSW16", "TSE", "TSEF"]), (40, ["FOLK", "ATSE", "ABSW16"])]:
                for gkey in ["1.5", "2"]:
                    for inst_id in range(1, 6):
                        for method in methods:
                            if (n, gkey, inst_id, method) not in done:
                                tasks.append((n, gkey, inst_id, method, a.budget))
            tasks.sort(key=lambda t: (t[3] in ("TSE", "BSW16", "TSEF"), t[0]))
            run(pool, job_large, tasks, a.out)


if __name__ == "__main__":
    main()
