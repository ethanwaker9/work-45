import argparse
import os
import time
from fractions import Fraction
from math import sqrt, ceil
from multiprocessing import Pool
from fpylll import IntegerMatrix, FPLLL
from common import RESULTS, GAMMAS, EPS, regions, gamma_value, append_jsonl, read_jsonl
from tubesparse.instances import random_lattice, bdd_instance, zn_instance, simplex_rows
from tubesparse.oracle import PromiseUSVPOracle, HelpfulUSVPOracle
from tubesparse.reductions import run_method
from tubesparse.lattice import lll_rows

EPSS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]


def job_eps(args):
    n, gkey, ga, inst_id, method, eps = args
    gamma = gamma_value(gkey)
    rows = random_lattice(n, 10, 1000 * n + inst_id)
    inst = bdd_instance(rows, ga / gamma, "adv", 7919 * inst_id + n)
    oracle = PromiseUSVPOracle(GAMMAS[gkey])
    res = run_method(method, inst, oracle, gamma, 104729 * inst_id + 31 * n, budget_factor=10, eps=eps)
    res.update({"part": "eps", "n": n, "gamma": gkey, "ga": ga, "inst": inst_id, "eps": eps})
    return res


def job_helpful(args):
    n, gkey, region, alpha, mode, inst_id, method = args
    gamma = gamma_value(gkey)
    rows = random_lattice(n, 10, 1000 * n + inst_id)
    inst = bdd_instance(rows, alpha, mode, 7919 * inst_id + n)
    oracle = HelpfulUSVPOracle(GAMMAS[gkey])
    res = run_method(method, inst, oracle, gamma, 104729 * inst_id + 31 * n, budget_factor=10, eps=EPS)
    res.update({"part": "helpful", "n": n, "gamma": gkey, "region": region, "mode": mode, "inst": inst_id})
    return res


def family_rows(fam, n, inst_id):
    if fam == "qary":
        return random_lattice(n, 10, 1000 * n + inst_id)
    if fam == "uniform":
        FPLLL.set_random_seed(2000 * n + inst_id)
        A = IntegerMatrix.random(n, "uniform", bits=6)
        return lll_rows([[int(A[i, j]) for j in range(n)] for i in range(n)])
    if fam == "zn":
        return zn_instance(n, 40 * n, 3, 300 * n + inst_id)
    if fam == "simplex":
        return simplex_rows(n, int(ceil(sqrt(n + 1))))
    raise ValueError(fam)


def job_family(args):
    fam, n, gkey, inst_id, method = args
    gamma = gamma_value(gkey)
    alpha = dict(regions(gamma))["R3b"]
    rows = family_rows(fam, n, inst_id)
    fi = ["qary", "uniform", "zn", "simplex"].index(fam)
    inst = bdd_instance(rows, alpha, "adv", 4001 * inst_id + 50021 * fi + n)
    oracle = PromiseUSVPOracle(GAMMAS[gkey])
    res = run_method(method, inst, oracle, gamma, 3301 * inst_id + 70001 * fi + n, budget_factor=10, eps=EPS)
    res.update({"part": "family", "family": fam, "n": n, "gamma": gkey, "inst": inst_id, "lam2": inst["lam2"]})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--parts", default="eps,helpful,family")
    ap.add_argument("--out", default=os.path.join(RESULTS, "extra.jsonl"))
    a = ap.parse_args()
    rs = read_jsonl(a.out)
    parts = a.parts.split(",")
    t0 = time.time()
    with Pool(a.workers) as pool:
        if "helpful" in parts:
            done = {(r["n"], r["gamma"], r["region"], r["mode"], r["inst"], r["method"]) for r in rs if r["part"] == "helpful"}
            tasks = []
            for n in [12, 16, 20, 24, 28]:
                for gkey in ["1.25", "1.5", "2", "3"]:
                    for region, alpha in regions(gamma_value(gkey)):
                        for mode in ["adv", "rand"]:
                            for i in range(1, 6):
                                for m in ["KAN", "LM09", "FOLK"]:
                                    if (n, gkey, region, mode, i, m) not in done:
                                        tasks.append((n, gkey, region, alpha, mode, i, m))
                                if mode == "adv" and region.startswith("R3") and (n, gkey, region, mode, i, "TSE") not in done:
                                    tasks.append((n, gkey, region, alpha, mode, i, "TSE"))
            for res in pool.imap_unordered(job_helpful, tasks):
                append_jsonl(a.out, res)
            print("helpful", len(tasks), round(time.time() - t0, 1), flush=True)
        if "eps" in parts:
            done = {(r["gamma"], round(r["ga"], 3), r["inst"], r["method"], r["eps"]) for r in rs if r["part"] == "eps"}
            tasks = [(20, g, ga, i, m, e) for g in ["1.5", "2"] for ga in [0.55, 0.60] for i in range(1, 6) for m in ["TSE", "ATSE"] for e in EPSS if (g, round(ga, 3), i, m, e) not in done]
            tasks.sort(key=lambda t: t[5])
            for res in pool.imap_unordered(job_eps, tasks):
                append_jsonl(a.out, res)
            print("eps", len(tasks), round(time.time() - t0, 1), flush=True)
        if "family" in parts:
            done = {(r["family"], r["n"], r["gamma"], r["inst"], r["method"]) for r in rs if r["part"] == "family"}
            tasks = []
            for fam in ["qary", "uniform", "zn", "simplex"]:
                for n in [16, 24]:
                    for g in ["1.5", "2"]:
                        insts = range(1, 6)
                        for i in insts:
                            for m in ["FOLK", "TSE", "ATSE", "BSW16", "ABSW16"]:
                                if (fam, n, g, i, m) not in done:
                                    tasks.append((fam, n, g, i, m))
            tasks.sort(key=lambda t: t[4] == "BSW16")
            for res in pool.imap_unordered(job_family, tasks):
                append_jsonl(a.out, res)
            print("family", len(tasks), round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
