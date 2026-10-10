import json
import os
import subprocess
import sys
from collections import defaultdict
from math import sqrt, floor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RESULTS, FIGURES, EPS, read_jsonl
from tubesparse.reductions import lm09_alpha, folk_alpha, sqrt2_alpha, guaranteed_alpha, one_call_radius
from tubesparse.isolation import tse_prime, bsw16_prime, tube_count_bound, tube_count_free, next_prime
from tubesparse.reductions import folk_k, folk_gamma1, one_call_radius
from common import GAMMAS, regions, gamma_value
import numpy as np

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.2, "lines.markersize": 3.5})

COLORS = {"KAN": "tab:gray", "LM09": "tab:blue", "FOLK": "tab:green", "BSW16": "tab:orange", "TSE": "tab:red"}
LABELS = {"KAN": "Embedding $M=d/\\sqrt{n}$", "LM09": "LM09", "FOLK": "Folklore", "BSW16": "BSW16", "TSE": "TSE (ours)"}


LLNCS = "--llncs" in sys.argv
OUT = os.path.join(os.path.dirname(FIGURES), "figures_llncs") if LLNCS else FIGURES
os.makedirs(OUT, exist_ok=True)


def wide(h):
    return (4.8, 1.3 * h) if LLNCS else (6.9, h)


def cols(k):
    return (2 if k == 4 else min(k, 3)) if LLNCS else k


FOLKNAME = "balanced" if LLNCS else "folklore"
EXACTNAME = "precise" if LLNCS else "exact"


def panels(k, h, sharey=False, sharex=False, ph=1.0):
    if LLNCS:
        fig, axes = plt.subplots(k, 1, figsize=(2.45, 0.45 + ph * k), sharex=sharex, gridspec_kw={"hspace": 0.55})
    else:
        fig, axes = plt.subplots(1, k, figsize=wide(h), sharey=sharey)
    return fig, list(np.atleast_1d(axes))


def ylab(fig, axes, text, shared):
    if LLNCS:
        if shared:
            for ax in axes[:-1]:
                ax.set_xlabel("")
        for ax in axes:
            ax.set_ylabel("")
        fig.supylabel(text, fontsize=8, x=-0.06)
    else:
        axes[0].set_ylabel(text)


def lcols(k):
    return 2 if LLNCS else k


def save(fig, name):
    eps = os.path.join(OUT, name + ".eps")
    fig.savefig(eps, format="eps", bbox_inches="tight")
    plt.close(fig)
    subprocess.run(["epstopdf", eps], check=True)


def fig_radius():
    gs = [1.0 + 0.004 * i for i in range(0, 1251)]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(3.3, 1.75), sharex=True, gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.08})
    ax.axhline(1 / sqrt(2), color="black", linewidth=0.7, linestyle=":")
    ax.plot(gs, [(1 - 1.0 / 20) / sqrt(2) for g in gs], color=COLORS["BSW16"], linewidth=1.2, label="BSW16 and TSE, $\\varepsilon=1/n$, $n=20$")
    ax.plot(gs, [(1 - EPS) / sqrt(2) for g in gs], color=COLORS["TSE"], linewidth=1.2, label="TSE, $\\varepsilon=0.1$")
    ax.plot(gs, [g * folk_alpha(g) for g in gs], color=COLORS["FOLK"], linewidth=1.2, label="one call, %s $M=k_\\gamma d$" % FOLKNAME)
    ax.plot(gs, [g * one_call_radius(g, 1.0) for g in gs], color=COLORS["LM09"], linewidth=0.9, linestyle="--", label="one call, $M=d$")
    ax.plot(gs, [0.5 for g in gs], color=COLORS["LM09"], linewidth=0.7, linestyle=":", label="bound of LM09 for $M=d$")
    bs = [1.0 + 0.25 * i for i in range(0, 21)]
    best = [max(b * one_call_radius(b, 0.005 * j) for j in range(1, 600)) for b in bs]
    ax.plot(bs, best, linestyle="none", marker="x", markersize=3.5, color="black", label="one call, best $M$")
    ax2.plot(gs, [g * one_call_radius(g, 1.0 / sqrt(20.0)) for g in gs], color=COLORS["KAN"], linewidth=1.0, label="one call, $M=d/\\sqrt{n}$, $n=20$")
    ax.set_ylim(0.48, 0.725)
    ax2.set_ylim(0.20, 0.245)
    ax.spines["bottom"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.tick_params(bottom=False)
    kw = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=6, linestyle="none", color="k", mec="k", mew=0.8, clip_on=False)
    ax.plot([0, 1], [0, 0], transform=ax.transAxes, **kw)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kw)
    ax2.set_xlabel("gap $\\gamma$")
    ax.set_ylabel("proven radius $\\gamma\\alpha$")
    ax.yaxis.set_label_coords(-0.12, 0.3)
    ax2.set_xlim(1, 6)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=5.8, columnspacing=0.8, handlelength=2.2)
    save(fig, "fig_radius")


def fig_tube():
    rs = read_jsonl(os.path.join(RESULTS, "tube.jsonl"))
    by = defaultdict(list)
    for r in rs:
        by[(r["eps"], r["n"])].append(r)
    fig, ax = plt.subplots(figsize=(2.45, 2.1) if LLNCS else (3.3, 1.9))
    marks = {0.05: "o", 0.1: "s", 0.2: "^"}
    ns = sorted({r["n"] for r in rs})
    for eps in sorted({r["eps"] for r in rs}):
        nn = sorted({r["n"] for r in rs if r["eps"] == eps})
        ax.plot(nn, [by[(eps, n)][0]["bound"] for n in nn], marker=marks.get(eps, "o"), color="tab:red", linestyle=":", label="tube lemma, $\\varepsilon=%g$" % eps)
    for eps in sorted({r["eps"] for r in rs}):
        nn = sorted({r["n"] for r in rs if r["eps"] == eps})
        ax.plot(nn, [max(x["count"] for x in by[(eps, n)]) for n in nn], marker=marks.get(eps, "o"), color="black", linestyle="-", label="measured, $\\varepsilon=%g$" % eps)
    for g, ls in [(1.0, "--"), (4.0, "-.")]:
        ax.plot(ns, [2 * g * n * n for n in ns], color="tab:orange", linestyle=ls, label="$2\\gamma n^2$, $\\gamma=%g$" % g)
    ax.set_yscale("log")
    ax.set_xlabel("dimension $n$")
    ax.set_ylabel("lattice points in the tube")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=lcols(3), frameon=False, fontsize=6, columnspacing=0.8, handlelength=2.2)
    save(fig, "fig_tube")


MARKS = {"KAN": ("v", 3.5, "full"), "LM09": ("s", 6.5, "none"), "FOLK": ("o", 3.5, "full"), "BSW16": ("D", 7.0, "none"), "TSE": ("o", 3.5, "full")}


def styled(ax, xs, ys, m, **kw):
    mk, ms, fill = MARKS[m]
    face = "none" if fill == "none" else COLORS[m]
    return ax.plot(xs, ys, marker=mk, markersize=ms, markerfacecolor=face, markeredgecolor=COLORS[m], color=COLORS[m], **kw)


def fig_sweep():
    rs = read_jsonl(os.path.join(RESULTS, "sweep.jsonl"))
    if not rs:
        return
    gk = sorted({r["gamma"] for r in rs}, key=float)
    fig, axes = panels(len(gk), 1.68, sharey=True, sharex=True)
    styles = {"BSW16": dict(color="#fdbf6f", linewidth=3.4), "TSE": dict(color=COLORS["TSE"], linewidth=1.2), "LM09": dict(color="#a6cee3", linewidth=3.4), "FOLK": dict(color=COLORS["FOLK"], linewidth=1.1), "KAN": dict(color=COLORS["KAN"], linewidth=1.1)}
    names = {"KAN": "one call, $M=d/\\sqrt{n}$", "LM09": "one call, $M=d$ (LM09)", "FOLK": "one call, " + FOLKNAME, "BSW16": "BSW16", "TSE": "TSE (ours)"}
    for ax, g in zip(axes, gk):
        gamma = float(g)
        for m in ["BSW16", "LM09", "KAN", "FOLK", "TSE"]:
            pts = defaultdict(list)
            for r in rs:
                if r["gamma"] == g and r["method"] == m:
                    pts[round(gamma * r["alpha"], 6)].append(r["success"])
            xs = sorted(pts)
            if xs:
                ax.plot(xs, [sum(pts[x]) / len(pts[x]) for x in xs], drawstyle="steps-mid", label=names[m], **styles[m])
        for m, k in [("KAN", 1.0 / sqrt(20.0)), ("LM09", 1.0), ("FOLK", folk_k(gamma))]:
            ax.axvline(gamma * one_call_radius(gamma, k), color=COLORS[m], linewidth=0.7, linestyle=":")
        ax.axvline((1 - EPS) / sqrt(2), color=COLORS["TSE"], linewidth=0.7, linestyle="--")
        ax.axvline((1 - 1.0 / 20) / sqrt(2), color=COLORS["BSW16"], linewidth=0.7, linestyle="--")
        ax.set_title(("$\\gamma=%s$, $n=20$" if LLNCS else "$\\gamma=%s$, $n=20$, ten lattices per radius") % g, fontsize=8)
        ax.set_xlabel("normalized radius $\\gamma\\alpha$")
        ax.set_xlim(0, 0.705)
        ax.set_ylim(-0.06, 1.08)
    ylab(fig, axes, "fraction solved", True)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=lcols(5), frameon=False, fontsize=6.5)
    save(fig, "fig_sweep")


def fig_calls():
    rs = read_jsonl(os.path.join(RESULTS, "bdd.jsonl"))
    fig, axes = panels(2, 1.75, sharex=True)
    gks = sorted({r["gamma"] for r in rs}, key=float)
    styles = {"1.25": "o", "1.5": "s", "2": "^", "3": "D"}
    hard = [r for r in rs if r["mode"] == "adv" and r["region"].startswith("R3")]
    for g in gks:
        by = defaultdict(list)
        bits = defaultdict(list)
        for r in hard:
            if r["gamma"] == g and r["method"] == "BSW16":
                by[r["n"]].append(r["calls"])
                bits[r["n"]].append(r["max_bits"])
        ns = sorted(by)
        axes[0].plot(ns, [sum(by[n]) / len(by[n]) for n in ns], marker=styles.get(g, "o"), color=COLORS["BSW16"], linestyle="--", label="BSW16, $\\gamma=%s$" % g)
        axes[1].plot(ns, [max(bits[n]) for n in ns], marker=styles.get(g, "o"), color=COLORS["BSW16"], linestyle="--")
    by = defaultdict(list)
    bits = defaultdict(list)
    for r in hard:
        if r["method"] == "TSE":
            by[r["n"]].append(r["calls"])
            bits[r["n"]].append(r["max_bits"])
    ns = sorted(by)
    axes[0].plot(ns, [sum(by[n]) / len(by[n]) for n in ns], marker="o", color=COLORS["TSE"], linestyle="-", label="TSE (ours), every $\\gamma$")
    axes[1].plot(ns, [max(bits[n]) for n in ns], marker="o", color=COLORS["TSE"], linestyle="-")
    axes[0].plot(ns, [tse_prime(n, EPS) for n in ns], color=COLORS["TSE"], linewidth=0.7, linestyle=":", label="prime of TSE")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("mean calls" if LLNCS else "mean oracle calls")
    axes[1].set_ylabel("bits per entry" if LLNCS else "max bits per oracle entry")
    for ax in axes:
        ax.set_xticks(ns)
        ax.set_xlabel("dimension $n$")
    if LLNCS:
        axes[0].set_xlabel("")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=lcols(3), frameon=False, fontsize=6.5)
    save(fig, "fig_calls")


def fig_gamma():
    rs = read_jsonl(os.path.join(RESULTS, "gamma.jsonl"))
    if not rs:
        return
    fig, ax = plt.subplots(figsize=(2.45, 2.1) if LLNCS else (3.3, 1.9))
    for n, mk in [(12, "o"), (16, "s")]:
        for m in ["TSE", "BSW16"]:
            by = defaultdict(list)
            for r in rs:
                if r["n"] == n and r["method"] == m and r["success"]:
                    by[r["gamma"]].append(r["calls"])
            gs = sorted(by)
            if gs:
                ax.plot(gs, [sum(by[g]) / len(by[g]) for g in gs], marker=mk, color=COLORS[m], linestyle="-" if m == "TSE" else "--", label="%s, $n=%d$" % ("TSE (ours)" if m == "TSE" else "BSW16", n))
    for n, ls in [(12, ":"), (16, (0, (1, 3)))]:
        gs = sorted({r["gamma"] for r in rs if r["n"] == n})
        ax.plot(gs, [tse_prime(n, EPS) for g in gs], color=COLORS["TSE"], linewidth=0.7, linestyle=ls, label="prime of TSE, $n=%d$" % n)
        ax.plot(gs, [bsw16_prime(n, g) for g in gs], color=COLORS["BSW16"], linewidth=0.7, linestyle=ls, label="prime of BSW16, $n=%d$" % n)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("gap $\\gamma$")
    ax.set_ylabel("mean oracle calls")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=6, columnspacing=0.8, handlelength=2.2)
    save(fig, "fig_gamma")


def fig_fold():
    rs = read_jsonl(os.path.join(RESULTS, "fold.jsonl"))
    if not rs:
        return
    gk = sorted({r["gamma"] for r in rs}, key=float)
    fig, axes = panels(len(gk), 1.9, sharey=True, sharex=True, ph=0.78)
    for ax, g in zip(axes, gk):
        sub = [r for r in rs if r["gamma"] == g]
        gas = sorted({round(r["ga"], 4) for r in sub})
        ks = sub[0]["ks"]
        grid = np.zeros((len(ks), len(gas)))
        cnt = np.zeros((len(ks), len(gas)))
        for r in sub:
            j = gas.index(round(r["ga"], 4))
            for i, ok in enumerate(r["ok"]):
                grid[i, j] += ok
                cnt[i, j] += 1
        frac = grid / np.maximum(cnt, 1)
        dx = gas[1] - gas[0]
        ax.pcolormesh([x - dx / 2 for x in gas] + [gas[-1] + dx / 2], [k * 10 ** (-1.1 / 44) for k in ks] + [ks[-1] * 10 ** (1.1 / 44)], frac, cmap="Greys", vmin=0, vmax=1, shading="flat")
        gamma = float(g) ** 1
        gam = gamma_value(g)
        ax.axvline(gam / folk_gamma1(gam), color="tab:red", linewidth=1.0, linestyle="--")
        ax.plot([gam / folk_gamma1(gam)], [folk_k(gam)], marker="*", color="tab:red", markersize=7)
        ax.set_yscale("log")
        ax.set_title("$\\gamma=%s$, $n=20$" % g, fontsize=8)
        ax.set_xlabel("$\\gamma\\alpha$")
    ylab(fig, axes, "scaling $M/d$", True)
    save(fig, "fig_fold")


def fig_calib():
    rs = [r for r in read_jsonl(os.path.join(RESULTS, "calib.jsonl")) if r["part"] == "rate"]
    if not rs:
        return
    ns = sorted({r["n"] for r in rs})
    fig, axes = panels(len(ns), 1.9, sharey=True, sharex=True, ph=0.8)
    for ax, n in zip(axes, ns):
        for cal, col, mk in [("TSE", COLORS["TSE"], "o"), ("BSW16", COLORS["BSW16"], "D")]:
            ps = sorted({r["p"] for r in rs if r["n"] == n and r["cal"] == cal})
            meas, pred = [], []
            for p in ps:
                v = [r for r in rs if r["n"] == n and r["cal"] == cal and r["p"] == p]
                meas.append(p * sum(r["succ"] for r in v) / sum(r["trials"] for r in v))
                pred.append(p * sum(r["pred"] for r in v) / len(v))
            ax.plot(ps, pred, color=col, linestyle="-", linewidth=0.9, label="%s scaling, %s" % ("TSE" if cal == "TSE" else "BSW16", EXACTNAME))
            ax.plot(ps, meas, color=col, linestyle="none", marker=mk, markerfacecolor="none", label="%s scaling, measured" % ("TSE" if cal == "TSE" else "BSW16"))
        g = gamma_value("2")
        alpha = dict(regions(g))["R3b"]
        thr = alpha * (sqrt(g * g + (g * g - 0.25) * (n - 1) ** 2) - 0.5)
        ax.axvline(thr, color=COLORS["BSW16"], linestyle=":", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_title("$n=%d$, $\\gamma=2$" % n, fontsize=8)
        ax.set_xlabel("prime $p$")
        ax.set_ylim(-0.05, 1.15)
    ylab(fig, axes, "$p\\,\\times$ success per trial", True)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=lcols(4), frameon=False, fontsize=6.5)
    save(fig, "fig_calib")


def fig_atse():
    at = read_jsonl(os.path.join(RESULTS, "atse.jsonl"))
    bdd = read_jsonl(os.path.join(RESULTS, "bdd.jsonl"))
    gam = read_jsonl(os.path.join(RESULTS, "gamma.jsonl"))
    fig, axes = panels(2, 2.0, ph=1.15)
    ax = axes[0]
    for m, col, ls, mk in [("TSE", COLORS["TSE"], "-", "o"), ("BSW16", COLORS["BSW16"], "--", "D"), ("ATSE", "tab:purple", "-", "s"), ("ABSW16", "tab:brown", "--", "^")]:
        by = defaultdict(list)
        for r in bdd:
            if r["method"] == m and r["mode"] == "adv" and r["region"].startswith("R3"):
                by[r["n"]].append(r["calls"])
        for r in at:
            if r["method"] == m and ((r["exp"] == "bdd" and r["mode"] == "adv" and r["region"].startswith("R3")) or r["exp"] == "large"):
                by[r["n"]].append(r["calls"])
        ns = sorted(by)
        ax.plot(ns, [sum(by[n]) / len(by[n]) for n in ns], color=col, linestyle=ls, marker=mk, label={"ATSE": "adaptive TSE", "ABSW16": "adaptive BSW16", "TSE": "TSE", "BSW16": "BSW16"}[m])
    ax.set_yscale("log")
    ax.set_xlabel("dimension $n$")
    ax.set_ylabel("mean oracle calls")
    ax.set_xticks([12, 16, 20, 24, 28, 32, 36, 40])
    ax = axes[1]
    for m, col, ls, mk, src in [("TSE", COLORS["TSE"], "-", "o", gam), ("BSW16", COLORS["BSW16"], "--", "D", gam), ("ATSE", "tab:purple", "-", "s", at), ("ABSW16", "tab:brown", "--", "^", at)]:
        by = defaultdict(list)
        for r in src:
            if r["method"] == m and r.get("exp", "gap") == "gap":
                by[float(r["gamma2"]) ** 0.5].append(r["calls"])
        gs = sorted(by)
        ax.plot(gs, [sum(by[g]) / len(by[g]) for g in gs], color=col, linestyle=ls, marker=mk)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("gap $\\gamma$, $n\\in\\{12,16\\}$")
    ylab(fig, axes, "mean oracle calls", False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=lcols(4), frameon=False, fontsize=6.5)
    save(fig, "fig_atse")


def fig_counts():
    ns = list(range(8, 257, 8))
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    for eps, ls in [(0.1, "-"), (0.5, "--")]:
        ax.plot(ns, [tube_count_bound(n, eps) for n in ns], color="tab:red", linestyle=ls, linewidth=0.9, label="$N(n,\\varepsilon)$, $\\varepsilon=%g$" % eps)
        ax.plot(ns, [tube_count_free(eps) for n in ns], color="tab:purple", linestyle=ls, linewidth=0.9, label="$N'(\\varepsilon)$, $\\varepsilon=%g$" % eps)
    ax.plot(ns, [2 * n * n for n in ns], color="tab:orange", linestyle=":", label="$2\\gamma n^2$, $\\gamma=1$")
    ax.plot([12, 16, 20, 24, 28], [4] * 5, color="black", marker="o", linestyle="none", label="measured $|\\mathcal{Y}^*|$")
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("dimension $n$")
    ax.set_ylabel("unwanted vectors")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=6)
    save(fig, "fig_counts")


def fig_lower():
    rs = [r for r in read_jsonl(os.path.join(RESULTS, "simplex.jsonl")) if r["part"] == "rate"]
    if not rs:
        return
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    for n, mk in [(6, "o"), (8, "s"), (10, "^"), (12, "D")]:
        v = sorted([r for r in rs if r["n"] == n], key=lambda r: r["p"])
        xs = [r["p"] / n for r in v]
        ax.plot(xs, [r["succ"] / r["trials"] * n for r in v], marker=mk, linestyle="none", markerfacecolor="none", color="black", label="measured, $n=%d$" % n)
        ax.plot(xs, [r["pred"] * n for r in v], color="tab:purple", linewidth=0.8, label=EXACTNAME if n == 6 else None)
        ax.plot(xs, [r["bound"] * n for r in v], color="tab:red", linewidth=0.8, linestyle="--", label="bound of the theorem" if n == 6 else None)
    ax.axhline(4.0 / (3.0 * np.e), color="gray", linestyle=":", linewidth=0.8, label="$4/(3e)$")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=6, columnspacing=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("$p/n$")
    ax.set_ylabel("$n\\,\\times$ success per trial")
    save(fig, "fig_lower")


def fig_dist():
    rs = read_jsonl(os.path.join(RESULTS, "calib.jsonl"))
    rep = [r for r in rs if r["part"] == "rep"]
    pred = {(r["n"], r["inst"]): r["TSE"] for r in rs if r["part"] == "pred" and r["gamma"] == "2" and r["region"] == "R3b"}
    if not rep:
        return
    fig, axes = panels(2, 1.9, sharey=True, ph=1.05)
    for ax, n in zip(axes, [12, 20]):
        for i, col in zip(range(1, 6), ["tab:red", "tab:blue", "tab:green", "tab:purple", "tab:brown"]):
            calls = sorted(r["calls"] for r in rep if r["n"] == n and r["inst"] == i)
            ys = [(k + 1) / len(calls) for k in range(len(calls))]
            ax.step(calls, ys, where="post", color=col, linewidth=1.0, label="lattice %d" % i if n == 12 else None)
        pr = pred[(n, 1)]
        q = pr["pi"] / pr["p"]
        xs = list(range(1, int(6 * pr["p"]), max(1, pr["p"] // 50)))
        ax.plot(xs, [1 - (1 - q) ** (x - 1) for x in xs], color="black", linestyle="--", linewidth=1.0, label=EXACTNAME + ", geometric" if n == 12 else None)
        ax.set_xlabel("oracle calls of TSE, $n=%d$, $p=%d$" % (n, pr["p"]))
        ax.set_xlim(0, 5 * pr["p"])
    ylab(fig, axes, "fraction of runs", False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=3 if LLNCS else 6, frameon=False, fontsize=6.5)
    save(fig, "fig_dist")


def fig_sweep_calls():
    rs = read_jsonl(os.path.join(RESULTS, "sweep.jsonl"))
    gk = sorted({r["gamma"] for r in rs}, key=float)
    fig, axes = plt.subplots(1, len(gk), figsize=wide(1.9), sharey=True)
    for ax, g in zip(axes, gk):
        gamma = float(g)
        for m, col, ls, mk, lab in [("TSE", COLORS["TSE"], "-", "o", "TSE"), ("BSW16", COLORS["BSW16"], "--", "D", "BSW16"), ("ATSE", "tab:purple", "-", "s", "adaptive TSE"), ("ABSW16", "tab:brown", "--", "^", "adaptive BSW16")]:
            pts = defaultdict(list)
            for r in rs:
                if r["gamma"] == g and r["method"] == m:
                    pts[r["alpha"]].append(r["calls"])
            xs = sorted(pts)
            ax.plot([gamma * x for x in xs], [sum(pts[x]) / len(pts[x]) for x in xs], color=col, linestyle=ls, marker=mk, markevery=5, markersize=3.2, linewidth=1.0, drawstyle="steps-mid", label=lab)
        ax.axvline(gamma * folk_alpha(gamma), color=COLORS["FOLK"], linewidth=0.7, linestyle=":")
        ax.set_yscale("log")
        ax.set_title("$\\gamma=%s$, $n=20$" % g, fontsize=8)
        ax.set_xlabel("$\\gamma\\alpha$")
    axes[0].set_ylabel("mean oracle calls")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=cols(4), frameon=False, fontsize=6.5)
    save(fig, "fig_sweep_calls")


def fig_mult():
    rs = read_jsonl(os.path.join(RESULTS, "mult.jsonl"))
    if not rs:
        return
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    marks = {"1.5": "o", "2": "s", "3": "^"}
    for g in ["1.5", "2", "3"]:
        sub = [r for r in rs if r["gamma"] == g]
        ax.plot([1 / r["kk"] for r in sub], [r["line"] for r in sub], marker=marks[g], linestyle="none", markerfacecolor="none", color="black", label="measured, $\\gamma=%s$" % g)
    ks = sorted({r["kk"] for r in rs})
    import math
    for g, ls in [("1.5", ":"), ("3", "--")]:
        sub = [r for r in rs if r["gamma"] == g and r["n"] == 20 and r["inst"] == 1]
        sub.sort(key=lambda r: r["kk"])
        ax.plot([1 / r["kk"] for r in sub], [2 * (math.ceil(r["upper"]) - 1) for r in sub], color="tab:red", linestyle=ls, linewidth=0.8, label="upper bound, $\\gamma=%s$" % g)
    ax.axvline(1 / sqrt(0.1), color=COLORS["TSE"], linewidth=0.7, linestyle="-.")
    for n in [12, 28]:
        ax.axvline(n - 1, color=COLORS["BSW16"], linewidth=0.7, linestyle="-.")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$d/M$")
    ax.set_ylabel("$|\\mathcal{Y}^*|$")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=6)
    save(fig, "fig_mult")


if __name__ == "__main__":
    fig_radius()
    fig_tube()
    fig_sweep()
    fig_calls()
    fig_gamma()
    fig_fold()
    fig_calib()
    fig_atse()
    fig_counts()
    fig_lower()
    fig_dist()
    fig_sweep_calls()
    fig_mult()
