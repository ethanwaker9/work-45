import random
from math import sqrt, isqrt
from fpylll import IntegerMatrix, FPLLL
from .lattice import lll_rows, shortest_vector, combine, norm2, add, scale


def random_lattice(n, bits, seed):
    FPLLL.set_random_seed(seed)
    A = IntegerMatrix.random(n, "qary", k=n // 2, bits=bits)
    return lll_rows([[int(A[i, j]) for j in range(n)] for i in range(n)])


def bdd_instance(rows, alpha, mode, seed, scale_factor=1 << 12, coeff=4):
    rng = random.Random(seed)
    n = len(rows)
    lam2, s = shortest_vector(rows)
    lam = sqrt(lam2)
    x = [rng.randint(-coeff, coeff) for _ in range(n)]
    c = combine(x, rows)
    if mode == "adv":
        direction = [v / lam for v in s]
    else:
        g = [rng.gauss(0.0, 1.0) for _ in range(n)]
        ng = sqrt(sum(v * v for v in g))
        direction = [v / ng for v in g]
    S = scale_factor
    e = [int(round(S * alpha * lam * v)) for v in direction]
    B = [scale(S, r) for r in rows]
    cS = scale(S, c)
    t = add(cS, e)
    return {
        "B": B,
        "t": t,
        "c": cS,
        "e": e,
        "d2": norm2(e),
        "lam2": lam2 * S * S,
        "n": n,
        "alpha": alpha,
        "mode": mode,
        "seed": seed,
    }


def random_unimodular(n, rounds, bound, seed):
    rng = random.Random(seed)
    U = [[int(i == j) for j in range(n)] for i in range(n)]
    for _ in range(rounds):
        i, j = rng.sample(range(n), 2)
        c = rng.randint(-bound, bound)
        U[i] = [a + c * b for a, b in zip(U[i], U[j])]
    perm = list(range(n))
    rng.shuffle(perm)
    return [U[k] for k in perm]


def zn_instance(n, rounds, bound, seed):
    return random_unimodular(n, rounds, bound, seed)


def primitive_count_zn(n, gamma2):
    from itertools import product
    r = isqrt(int(gamma2))
    cnt = 0
    for v in product(range(-r, r + 1), repeat=n):
        s = sum(x * x for x in v)
        if 0 < s <= gamma2:
            g = 0
            for x in v:
                a, b = abs(x), g
                while b:
                    a, b = b, a % b
                g = a
            if g == 1:
                cnt += 1
    return cnt // 2


def simplex_rows(n, b):
    return [[(b - 1) + (n if j == i else 0) for j in range(n)] for i in range(n)]


def simplex_bdd_instance(n, gamma2, eps, Q=100000):
    from fractions import Fraction
    gamma = float(gamma2) ** 0.5
    tau0 = (1.0 - eps) * sqrt(n) / gamma
    P = int(Q * tau0)
    tau = Fraction(P, Q)
    K = 2
    while K * tau < sqrt(n + 1) + 0.5:
        K += 1
    b = int(round(float(K * tau)))
    rows = [[Q * x for x in r] for r in simplex_rows(n, b)]
    t = [P] * n
    return {"B": rows, "t": t, "c": [0] * n, "e": list(t), "d2": n * P * P, "lam2": 2 * n * n * Q * Q, "n": n, "K": K, "b": b, "tau": float(tau), "Q": Q}
