import json
import os
from collections import defaultdict
from statistics import mean, median
from common import RESULTS, read_jsonl

METHODS = ["KAN", "LM09", "FOLK", "BSW16", "TSE"]


def main_summary(rs):
    out = {}
    for m in METHODS:
        sub = [r for r in rs if r["method"] == m]
        if not sub:
            continue
        hard = [r for r in sub if r["mode"] == "adv" and r["region"].startswith("R3")]
        out[m] = {
            "instances": len(sub),
            "solved": sum(r["success"] for r in sub),
            "mean_calls": mean(r["calls"] for r in sub),
            "mean_calls_hard": mean(r["calls"] for r in hard) if hard else None,
            "solved_hard": sum(r["success"] for r in hard),
            "hard": len(hard),
            "mean_seconds": mean(r["seconds"] for r in sub),
            "mean_seconds_hard": mean(r["seconds"] for r in hard) if hard else None,
            "mean_kib": mean(r["peak_bytes"] for r in sub) / 1024.0,
            "mean_kib_hard": mean(r["peak_bytes"] for r in hard) / 1024.0 if hard else None,
            "mean_bits": mean(r["max_bits"] for r in sub),
            "mean_bits_hard": mean(r["max_bits"] for r in hard) if hard else None,
            "max_bits": max(r["max_bits"] for r in sub),
            "peak_kib": max(r["peak_bytes"] for r in sub) / 1024.0,
            "median_peak_kib": median(r["peak_bytes"] for r in sub) / 1024.0,
        }
    return out


def paired_summary(rs):
    by = defaultdict(dict)
    for r in rs:
        by[(r["n"], r["gamma"], r["region"], r["mode"], r["inst"])][r["method"]] = r
    out = {}
    for m in METHODS:
        keys = [k for k, d in by.items() if m in d and "TSE" in d and d[m]["success"]]
        if not keys:
            continue
        out[m] = {
            "solved": len(keys),
            "tse_solved": sum(by[k]["TSE"]["success"] for k in keys),
            "calls_ratio": mean(by[k][m]["calls"] for k in keys) / mean(by[k]["TSE"]["calls"] for k in keys),
            "time_ratio": mean(by[k][m]["seconds"] for k in keys) / mean(by[k]["TSE"]["seconds"] for k in keys),
            "memory_ratio": mean(by[k][m]["peak_bytes"] for k in keys) / mean(by[k]["TSE"]["peak_bytes"] for k in keys),
            "bits_diff": mean(by[k][m]["max_bits"] for k in keys) - mean(by[k]["TSE"]["max_bits"] for k in keys),
        }
    return out


def region_summary(rs):
    out = defaultdict(dict)
    for m in METHODS:
        for reg in ["R1", "R2", "R3a", "R3b"]:
            for mode in ["adv", "rand"]:
                sub = [r for r in rs if r["method"] == m and r["region"] == reg and r["mode"] == mode]
                if sub:
                    out[m]["%s_%s" % (reg, mode)] = {"n": len(sub), "solved": sum(r["success"] for r in sub), "mean_calls": mean(r["calls"] for r in sub), "mean_seconds": mean(r["seconds"] for r in sub)}
    return out


def per_dim(rs):
    out = defaultdict(dict)
    for m in ["TSE", "BSW16"]:
        for n in sorted({r["n"] for r in rs}):
            sub = [r for r in rs if r["method"] == m and r["n"] == n and r["mode"] == "adv" and r["region"].startswith("R3")]
            if sub:
                out[m][n] = {"mean_calls": mean(r["calls"] for r in sub), "mean_seconds": mean(r["seconds"] for r in sub), "max_bits": max(r["max_bits"] for r in sub), "prime": max(r["prime"] for r in sub), "solved": sum(r["success"] for r in sub), "count": len(sub)}
    return out


def zsvp_summary(rs):
    out = []
    keys = sorted({(r["n"], r["gamma2"]) for r in rs})
    for n, g2 in keys:
        sub = [r for r in rs if r["n"] == n and r["gamma2"] == g2]
        out.append({"n": n, "gamma2": g2, "A": sub[0]["A"], "prime": sub[0]["prime"], "family": sub[0]["family_size"], "det_mean": mean(r["det_calls"] for r in sub), "det_max": max(r["det_calls"] for r in sub), "rnd_mean": mean(r["rnd_calls"] for r in sub), "all_unit": all(r["det_norm"] == 1 for r in sub), "count": len(sub)})
    return out


def det_summary(rs, bdd):
    tse = {(r["n"], r["gamma"], r["region"], r["inst"]): r for r in bdd if r["method"] == "TSE" and r["mode"] == "adv"}
    bsw = {(r["n"], r["gamma"], r["region"], r["inst"]): r for r in bdd if r["method"] == "BSW16" and r["mode"] == "adv"}
    out = {}
    for n in sorted({r["n"] for r in rs}):
        sub = [r for r in rs if r["n"] == n]
        keys = [(r["n"], r["gamma"], r["region"], r["inst"]) for r in sub]
        out[n] = {
            "count": len(sub),
            "solved": sum(r["success"] for r in sub),
            "prime": sub[0]["prime"],
            "mean_calls": mean(r["calls"] for r in sub),
            "median_calls": median(r["calls"] for r in sub),
            "max_calls": max(r["calls"] for r in sub),
            "mean_seconds": mean(r["seconds"] for r in sub),
            "max_bits": max(r["max_bits"] for r in sub),
            "mean_kib": mean(r["peak_bytes"] for r in sub) / 1024.0,
            "tse_calls": mean(tse[k]["calls"] for k in keys if k in tse),
            "bsw16_calls": mean(bsw[k]["calls"] for k in keys if k in bsw),
        }
    return out


def gamma_summary(rs):
    out = defaultdict(dict)
    for r in rs:
        key = "%s_%d" % (r["method"], r["n"])
        out[key].setdefault(r["gamma2"], []).append(r["calls"])
    return {k: {g: mean(v) for g, v in d.items()} for k, d in out.items()}


def sweep_summary(rs):
    out = defaultdict(dict)
    for r in rs:
        k = "%s_%s" % (r["method"], r["gamma"])
        out[k].setdefault(round(r["alpha"], 5), []).append(r["success"])
    return {k: {a: sum(v) / len(v) for a, v in d.items()} for k, d in out.items()}


def method_summary(rs, key):
    out = defaultdict(dict)
    for r in rs:
        out[r["method"]].setdefault(str(key(r)), []).append(r)
    return {m: {k: {"runs": len(v), "solved": sum(x["success"] for x in v), "mean_calls": mean(x["calls"] for x in v), "max_calls": max(x["calls"] for x in v), "mean_seconds": mean(x["seconds"] for x in v)} for k, v in d.items()} for m, d in out.items()}


def zsvp_extra(thin, bgps):
    out = {}
    for r in thin:
        k = "%d_%d" % (r["n"], r["gamma2"])
        out.setdefault(k, {"thin": [], "pencil": [], "random": [], "bgps": []})
        for m in ["thin", "pencil", "random"]:
            out[k][m].append(r[m]["calls"])
    for r in bgps:
        k = "%d_%d" % (r["n"], r["gamma2"])
        if k in out:
            out[k]["bgps"].append(r["calls"])
    return {k: {m: mean(v) for m, v in d.items() if v} for k, d in out.items()}


def calib_summary(rs):
    rate = [r for r in rs if r["part"] == "rate"]
    cells = defaultdict(lambda: [0, 0, 0.0, 0])
    for r in rate:
        c = cells[(r["n"], r["cal"], r["p"])]
        c[0] += r["succ"]
        c[1] += r["trials"]
        c[2] += r["pred"]
        c[3] += 1
    dev = [abs(s / t - p / c) / (p / c) for s, t, p, c in cells.values() if p > 0]
    zero = sum(s for s, t, p, c in cells.values() if p == 0)
    rep = [r for r in rs if r["part"] == "rep"]
    return {"cells": len(cells), "mean_relative_deviation": mean(dev), "successes_where_zero_predicted": zero, "repeated_runs": {str(n): mean(r["calls"] for r in rep if r["n"] == n) for n in sorted({r["n"] for r in rep})}}


def load(name):
    return read_jsonl(os.path.join(RESULTS, name))


if __name__ == "__main__":
    bdd = read_jsonl(os.path.join(RESULTS, "bdd.jsonl"))
    summary = {
        "main": main_summary(bdd),
        "paired": paired_summary(bdd),
        "regions": region_summary(bdd),
        "per_dim": per_dim(bdd),
        "zsvp": zsvp_summary(read_jsonl(os.path.join(RESULTS, "zsvp.jsonl"))),
        "gamma": gamma_summary(read_jsonl(os.path.join(RESULTS, "gamma.jsonl"))),
        "sweep": sweep_summary(read_jsonl(os.path.join(RESULTS, "sweep.jsonl"))),
        "det": det_summary(read_jsonl(os.path.join(RESULTS, "det.jsonl")), bdd),
        "dets": method_summary(load("dets.jsonl"), lambda r: r["n"]),
        "atse_main": method_summary([r for r in load("atse.jsonl") if r["exp"] == "bdd"], lambda r: r["n"]),
        "atse_gap": method_summary([r for r in load("atse.jsonl") if r["exp"] == "gap"], lambda r: (r["n"], r["gamma2"])),
        "large": method_summary([r for r in load("atse.jsonl") if r["exp"] == "large"], lambda r: r["n"]),
        "zsvp_families": zsvp_extra(load("thin.jsonl"), load("bgps.jsonl")),
        "calib": calib_summary(load("calib.jsonl")),
        "eps": method_summary([r for r in load("extra.jsonl") if r["part"] == "eps"], lambda r: r["eps"]),
        "helpful": method_summary([r for r in load("extra.jsonl") if r["part"] == "helpful"], lambda r: (r["region"], r["mode"])),
        "families": method_summary([r for r in load("extra.jsonl") if r["part"] == "family"], lambda r: (r["family"], r["n"])),
        "beyond": method_summary([r for r in load("beyond.jsonl") if r["method"] != "Y"], lambda r: r["ga"]),
    }
    with open(os.path.join(RESULTS, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    print(json.dumps(summary["main"], indent=1))
    print(json.dumps(summary["paired"], indent=1))
