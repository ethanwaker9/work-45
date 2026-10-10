from fractions import Fraction
from math import isqrt
from fpylll import IntegerMatrix, LLL, GSO, Enumeration, FPLLL

FPLLL.set_precision(160)


def norm2(v):
    return sum(x * x for x in v)


def bits_of(rows):
    m = 0
    for r in rows:
        for x in r:
            b = abs(int(x)).bit_length()
            if b > m:
                m = b
    return m


def combine(coeffs, rows):
    n = len(rows[0])
    out = [0] * n
    for c, r in zip(coeffs, rows):
        if c:
            for j in range(n):
                out[j] += c * r[j]
    return out


def add(u, v):
    return [a + b for a, b in zip(u, v)]


def sub(u, v):
    return [a - b for a, b in zip(u, v)]


def scale(c, v):
    return [c * a for a in v]


def lll_rows(rows):
    A = IntegerMatrix.from_matrix(rows)
    LLL.reduction(A)
    out = []
    for i in range(A.nrows):
        r = [int(A[i, j]) for j in range(A.ncols)]
        if any(r):
            out.append(r)
    return out


def shortest_list(rows, k, radius2=None):
    A = IntegerMatrix.from_matrix(rows)
    LLL.reduction(A)
    red = [[int(A[i, j]) for j in range(A.ncols)] for i in range(A.nrows)]
    M = GSO.Mat(A, float_type="mpfr")
    M.update_gso()
    n = A.nrows
    if radius2 is None:
        radius2 = norm2(red[0])
    E = Enumeration(M, nr_solutions=k)
    sols = E.enumerate(0, n, float(radius2) * (1.0 + 1e-9) + 1.0, 0)
    out = []
    for _, x in sols:
        coeffs = [int(round(c)) for c in x]
        v = combine(coeffs, red)
        if any(v):
            out.append((norm2(v), v))
    out.sort(key=lambda t: t[0])
    return out, red


def shortest_vector(rows):
    sols, red = shortest_list(rows, 1)
    if not sols:
        v = red[0]
        return norm2(v), v
    return sols[0]


def parallel(u, v):
    i = next((j for j, x in enumerate(v) if x != 0), None)
    if i is None:
        return not any(u)
    for j in range(len(u)):
        if u[j] * v[i] != v[j] * u[i]:
            return False
    return True


def isqrt_ceil(x):
    r = isqrt(x)
    return r if r * r == x else r + 1


def rational_inverse(rows):
    n = len(rows)
    A = [[Fraction(x) for x in r] + [Fraction(int(i == j)) for j in range(n)] for i, r in enumerate(rows)]
    for c in range(n):
        p = next(r for r in range(c, n) if A[r][c] != 0)
        A[c], A[p] = A[p], A[c]
        piv = A[c][c]
        A[c] = [x / piv for x in A[c]]
        for r in range(n):
            if r != c and A[r][c] != 0:
                f = A[r][c]
                A[r] = [a - f * b for a, b in zip(A[r], A[c])]
    return [row[n:] for row in A]


def coords(v, inv):
    n = len(inv)
    return [sum(Fraction(v[i]) * inv[i][j] for i in range(n)) for j in range(n)]


def in_lattice(v, inv):
    return all(c.denominator == 1 for c in coords(v, inv))
