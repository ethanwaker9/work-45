import argparse
import os
import random
import time
from math import sqrt
from multiprocessing import Pool
from common import RESULTS, GAMMAS, EPS, regions, gamma_value, append_jsonl, read_jsonl
from tubesparse.instances import random_lattice, bdd_instance
from tubesparse.oracle import PromiseUSVPOracle
from tubesparse.reductions import sparsified_trial
from tubesparse.isolation import tse_prime, bsw16_prime
from tubesparse.lattice import rational_inverse
from tubesparse.tube import unwanted_exact, residue_lines, avoid_probability

PRIMES = [3, 5, 7, 11, 13, 17, 19, 23, 29, 37, 47, 59, 71]


def make(n, gkey, region, inst_id):
    alpha = dict(regions(gamma_value(gkey)))[region]
    rows = random_lattice(n, 10, 1000 * n + inst_id)
    return bdd_instance(rows, alpha, "adv", 7919 * inst_id + n)


def scaling(inst, cal):
    n = inst["n"]
    d = sqrt(inst["d2"])
    lam = sqrt(inst["lam2"])
    if cal == "TSE":
        return max(1, int(round(sqrt(EPS) * d))), 0.0
    return max(1, int(round(d / (n - 1)))), (n + 1) * lam / sqrt(2.0)


def job_rate(args):
    n, gkey, inst_id, cal, p, trials = args
    inst = make(n, gkey, "R3b", inst_id)
    m, radius = scaling(inst, cal)
    Y = unwanted_exact(inst["B"], inst["e"], m, GAMMAS[gkey])
    inv = rational_inverse(inst["B"])
    zero, lines = residue_lines(Y, inv, p)
    pred = 0.0 if zero else float(avoid_probability(lines, n, p)) / p
    oracle = PromiseUSVPOracle(GAMMAS[gkey])
    rng = random.Random(1000003 * inst_id + 101 * p + n + (7 if cal == "TSE" else 13))
    succ = sum(sparsified_trial(inst, oracle, rng, p, m, radius) for _ in range(trials))
    return {"part": "rate", "n": n, "gamma": gkey, "inst": inst_id, "cal": cal, "p": p, "trials": trials, "succ": succ, "pred": pred, "Y": len(Y), "lines": len(lines), "zero": zero}


def job_pred(args):
    n, gkey, region, inst_id = args
    inst = make(n, gkey, region, inst_id)
    out = {"part": "pred", "n": n, "gamma": gkey, "region": region, "inst": inst_id}
    inv = rational_inverse(inst["B"])
    for cal, p in [("TSE", tse_prime(n, EPS)), ("BSW16", bsw16_prime(n, gamma_value(gkey)))]:
        m, _ = scaling(inst, cal)
        Y = unwanted_exact(inst["B"], inst["e"], m, GAMMAS[gkey])
        zero, lines = residue_lines(Y, inv, p)
        pi = 0.0 if zero else float(avoid_probability(lines, n, p))
        out[cal] = {"p": p, "Y": len(Y), "lines": len(lines), "pi": pi, "expected_trials": (p / pi) if pi > 0 else None}
    return out


def job_rep(args):
    n, gkey, inst_id, seed = args
    from tubesparse.reductions import run_method
    inst = make(n, gkey, "R3b", inst_id)
    oracle = PromiseUSVPOracle(GAMMAS[gkey])
    res = run_method("TSE", inst, oracle, gamma_value(gkey), 999331 * seed + 17 * inst_id + n, budget_factor=10, eps=EPS)
    res.update({"part": "rep", "n": n, "gamma": gkey, "inst": inst_id, "seed": seed})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--factor", type=int, default=50)
    ap.add_argument("--out", default=os.path.join(RESULTS, "calib.jsonl"))
    a = ap.parse_args()
    rs = read_jsonl(a.out)
    done_rate = {(r["n"], r["inst"], r["cal"], r["p"]) for r in rs if r["part"] == "rate"}
    done_pred = {(r["n"], r["gamma"], r["region"], r["inst"]) for r in rs if r["part"] == "pred"}
    rate_tasks = [(n, "2", i, cal, p, a.factor * p) for n in [12, 20, 28] for i in range(1, 6) for cal in ["TSE", "BSW16"] for p in PRIMES if (n, i, cal, p) not in done_rate]
    pred_tasks = [(n, g, reg, i) for n in [12, 16, 20, 24, 28] for g in ["1.25", "1.5", "2", "3"] for reg in ["R3a", "R3b"] for i in range(1, 6) if (n, g, reg, i) not in done_pred]
    rate_tasks.sort(key=lambda t: -t[4] * t[0])
    done_rep = {(r["n"], r["inst"], r["seed"]) for r in rs if r["part"] == "rep"}
    rep_tasks = [(n, "2", i, s) for n in [12, 20] for i in range(1, 6) for s in range(1, 41) if (n, i, s) not in done_rep]
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(job_pred, pred_tasks)):
            append_jsonl(a.out, res)
        print("pred", len(pred_tasks), round(time.time() - t0, 1), flush=True)
        for i, res in enumerate(pool.imap_unordered(job_rate, rate_tasks)):
            append_jsonl(a.out, res)
            if (i + 1) % 25 == 0:
                print("rate", i + 1, len(rate_tasks), round(time.time() - t0, 1), flush=True)
        for i, res in enumerate(pool.imap_unordered(job_rep, rep_tasks)):
            append_jsonl(a.out, res)
        print("rep", len(rep_tasks), round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
