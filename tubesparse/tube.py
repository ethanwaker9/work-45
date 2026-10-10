from math import sqrt, ceil, floor
from fpylll import IntegerMatrix, LLL, GSO, Enumeration
from .lattice import combine


def segment_distance(y, direction, half_length):
    s = sum(a * b for a, b in zip(y, direction))
    s = max(-half_length, min(half_length, s))
    return sqrt(sum((a - s * b) ** 2 for a, b in zip(y, direction)))


def tube_points(rows, direction, half_length, radius, spacing, cap=None):
    A = IntegerMatrix.from_matrix(rows)
    LLL.reduction(A)
    red = [[int(A[i, j]) for j in range(A.ncols)] for i in range(A.nrows)]
    M = GSO.Mat(A, float_type="mpfr")
    M.update_gso()
    n = A.nrows
    if cap is None:
        cap = 8 * n + 16
    steps = int(ceil(2 * half_length / spacing))
    ball = sqrt(radius * radius + (spacing / 2.0) ** 2)
    found = {}
    for i in range(steps + 1):
        s = -half_length + i * (2 * half_length) / max(steps, 1)
        center = tuple(s * b for b in direction)
        target = M.from_canonical(center)
        E = Enumeration(M, nr_solutions=cap)
        try:
            sols = E.enumerate(0, n, ball * ball, 0, target=target)
        except Exception:
            sols = []
        for _, x in sols:
            v = combine([int(round(c)) for c in x], red)
            key = tuple(v)
            if key not in found and any(v):
                if segment_distance(v, direction, half_length) < radius:
                    found[key] = v
    return list(found.values())


def tube_bound(n, half_length, radius, lam):
    eta = 1.0 - 2.0 * radius * radius / (lam * lam)
    if eta <= 0:
        return float("inf")
    return n * (int(floor((2 * half_length + 2 * radius) / (lam * sqrt(eta)))) + 1)


def ball_union_bound(n, gamma):
    return 2 * gamma * n * n


def unwanted_exact(rows, e, M, gamma2, cap=4000):
    from fractions import Fraction
    from .lattice import norm2
    A = IntegerMatrix.from_matrix(rows)
    LLL.reduction(A)
    red = [[int(A[i, j]) for j in range(A.ncols)] for i in range(A.nrows)]
    G = GSO.Mat(A, float_type="mpfr")
    G.update_gso()
    n = A.nrows
    d2 = norm2(e)
    r2 = Fraction(gamma2) * (d2 + M * M)
    found = {}
    m = 1
    while m * m * M * M < r2:
        bound = r2 - m * m * M * M
        center = [-m * x for x in e]
        target = G.from_canonical(center)
        E = Enumeration(G, nr_solutions=cap)
        try:
            sols = E.enumerate(0, n, float(bound) * (1.0 + 1e-9) + 1.0, 0, target=target)
        except Exception:
            sols = []
        for _, x in sols:
            y = combine([int(round(c)) for c in x], red)
            if any(y) and norm2([a + m * b for a, b in zip(y, e)]) + m * m * M * M < r2:
                found[tuple(y)] = m
                found[tuple(-a for a in y)] = -m
        m += 1
    return [list(k) for k in found]


def residue_lines(ys, inv, p):
    from .lattice import coords
    lines = set()
    zero = False
    for y in ys:
        c = [int(x) % p for x in coords(y, inv)]
        if not any(c):
            zero = True
            continue
        i = next(j for j, x in enumerate(c) if x)
        f = pow(c[i], p - 2, p)
        lines.add(tuple((x * f) % p for x in c))
    return zero, [list(v) for v in lines]


def rank_mod(vs, p):
    M = [list(v) for v in vs]
    r = 0
    if not M:
        return 0
    cols = len(M[0])
    for c in range(cols):
        piv = next((i for i in range(r, len(M)) if M[i][c] % p), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        f = pow(M[r][c], p - 2, p)
        M[r] = [(x * f) % p for x in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c] % p:
                g = M[i][c]
                M[i] = [(a - g * b) % p for a, b in zip(M[i], M[r])]
        r += 1
    return r


def avoid_probability(lines, n, p):
    from itertools import combinations
    from fractions import Fraction
    k = len(lines)
    total = 0
    for size in range(k + 1):
        for S in combinations(range(k), size):
            rk = rank_mod([lines[i] for i in S], p)
            total += (-1) ** size * (p ** (n - rk) - 1)
    return Fraction(total, p ** n - 1)
