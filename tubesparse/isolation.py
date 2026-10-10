from math import ceil, sqrt


def is_prime(m):
    if m < 2:
        return False
    if m % 2 == 0:
        return m == 2
    f = 3
    while f * f <= m:
        if m % f == 0:
            return False
        f += 2
    return True


def next_prime(m):
    m = max(2, int(m))
    while not is_prime(m):
        m += 1
    return m


def tube_count_bound(n, eps):
    k = int(1.5 * (1.0 / eps + 1.0 / sqrt(eps))) + 1
    return n * k


def tse_prime(n, eps):
    return next_prime(2 * tube_count_bound(n, eps))


def bsw16_prime(n, gamma):
    return next_prime(int(4 * gamma * n * n) + 1)


def sparsified_basis(rows, z, p):
    n = len(rows)
    j = next(i for i in range(n) if z[i] % p != 0)
    inv = pow(z[j] % p, p - 2, p)
    bj = rows[j]
    out = []
    for i in range(n):
        if i == j:
            continue
        c = (z[i] * inv) % p
        out.append([a - c * b for a, b in zip(rows[i], bj)])
    out.append([p * b for b in bj])
    return out


def moment(alpha, n, p):
    v = [1] * n
    for i in range(1, n):
        v[i] = (v[i - 1] * alpha) % p
    return v


def pencil_family(n, p):
    for a in range(p):
        ma = moment(a, n, p)
        yield ma
    for a in range(p):
        ma = moment(a, n, p)
        for a2 in range(p):
            if a2 == a:
                continue
            mb = moment(a2, n, p)
            for beta in range(p):
                yield [(x + beta * y) % p for x, y in zip(ma, mb)]


def pencil_size(p):
    return p + p * (p - 1) * p


def isolation_prime(n, a):
    return next_prime(2 * a * (n - 1) + 2 * n)


def tube_count_free(eps):
    slices = int(2.6 * (1.0 / eps + 1.0 / sqrt(eps))) + 1
    per_slice = int(ceil(1.64 / eps))
    return slices * per_slice


def tube_count_min(n, eps):
    return min(tube_count_bound(n, eps), tube_count_free(eps))


def h_eps(eps):
    return (1.0 + 1.0 / sqrt(eps)) / sqrt(2.0)


def tse_prime_min(n, eps):
    return next_prime(max(2 * tube_count_min(n, eps), int(h_eps(eps)) + 2))


def thin_pencil_family(n, p):
    for a2 in range(n):
        mb = moment(a2, n, p)
        for a in range(p):
            if a == a2:
                continue
            ma = moment(a, n, p)
            for beta in range(p):
                yield [(x + beta * y) % p for x, y in zip(ma, mb)]


def thin_pencil_size(n, p):
    return n * (p - 1) * p


def thin_isolation_prime(n, a):
    return next_prime(max(a * (n - 1) + 2, n))
