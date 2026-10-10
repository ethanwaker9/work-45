import json
import os
import sys
from fractions import Fraction
from math import sqrt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tubesparse.reductions import lm09_alpha, folk_alpha, sqrt2_alpha

RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

GAMMAS = {"1.25": Fraction(25, 16), "1.5": Fraction(9, 4), "2": Fraction(4), "3": Fraction(9)}
EPS = 0.1


def gamma_value(key):
    return sqrt(float(GAMMAS[key]))


def regions(gamma, eps=EPS):
    a_lm = lm09_alpha(gamma)
    a_f = folk_alpha(gamma)
    a_t = (1.0 - eps) * sqrt2_alpha(gamma)
    out = [("R1", 0.9 * a_lm)]
    if a_f > a_lm + 1e-9:
        out.append(("R2", 0.5 * (a_lm + a_f)))
    out.append(("R3a", a_f + 0.25 * (a_t - a_f)))
    out.append(("R3b", a_f + 0.75 * (a_t - a_f)))
    return out


def append_jsonl(path, rec):
    with open(path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]
