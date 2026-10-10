import random
import time
import tracemalloc
from math import sqrt, floor, ceil, log
from .lattice import norm2, add, sub, scale, combine, bits_of
from .isolation import tse_prime_min, sparsified_basis, tse_prime, bsw16_prime, pencil_family, isolation_prime, tube_count_bound, next_prime, tube_count_min, h_eps, thin_pencil_family, thin_isolation_prime, moment


def embedding(rows, target, m):
    return [r + [0] for r in rows] + [list(target) + [m]]


def extract(v, target, m):
    if v is None:
        return None
    if v[-1] == m:
        return sub(target, v[:-1])
    if v[-1] == -m:
        return add(target, v[:-1])
    return None


def folk_k(gamma):
    a = floor(gamma)
    return sqrt(4.0 * gamma * gamma - 1.0) / (2 * a + 1)


def folk_gamma1(gamma):
    a = floor(gamma)
    return (2.0 * gamma * gamma + 2.0 * a * (a + 1)) / (2 * a + 1)


def lm09_alpha(gamma):
    return 1.0 / (2.0 * gamma)


def folk_alpha(gamma):
    return 1.0 / folk_gamma1(gamma)


def sqrt2_alpha(gamma):
    return 1.0 / (sqrt(2.0) * gamma)


def one_call_radius(gamma, k):
    c = gamma * gamma * (1.0 + k * k)
    beta = sqrt(c)
    m = 1
    while m * k < sqrt(c):
        beta = max(beta, m + sqrt(c - m * m * k * k))
        m += 1
    return 1.0 / beta


def embedding_slack(alpha, gamma, k):
    inv = 1.0 / alpha
    rhs = gamma * gamma * (1.0 + k * k)
    worst = inv * inv - rhs
    mmax = int(ceil(gamma * sqrt(1.0 + k * k) / k)) + 1
    for m in range(1, mmax + 1):
        base = max(0.0, inv - m)
        val = base * base + m * m * k * k - rhs
        if val < worst:
            worst = val
    return worst


def optimal_k(alpha, gamma, grid=4000, kmax=3.0):
    best_k, best_s = 1.0, -float("inf")
    for i in range(1, grid + 1):
        k = kmax * i / grid
        sl = embedding_slack(alpha, gamma, k)
        if sl > best_s:
            best_k, best_s = k, sl
    return best_k, best_s


def guaranteed_alpha(gamma, k=None, lo=1e-4, hi=0.75, iters=60):
    def ok(a):
        if k is None:
            return optimal_k(a, gamma, grid=600)[1] >= 0.0
        return embedding_slack(a, gamma, k) >= 0.0
    if not ok(lo):
        return 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def single_call(inst, oracle, mfactor):
    d = sqrt(inst["d2"])
    m = max(1, int(round(mfactor * d)))
    v, _ = oracle(embedding(inst["B"], inst["t"], m))
    c = extract(v, inst["t"], m)
    return c is not None and norm2(sub(inst["t"], c)) == inst["d2"]


def shifted_target(rows, t, u, p, radius):
    w = combine(u, rows)
    if radius <= 0:
        return w
    ts = add(t, w)
    b1 = rows[0]
    nb1 = sqrt(norm2(b1))
    need = radius + sqrt(norm2(ts))
    h = max(0, ceil(need / (p * nb1)))
    w = add(w, scale(h * p, b1))
    return w


def sparsified_trial(inst, oracle, rng, p, m, radius):
    n = inst["n"]
    rows = inst["B"]
    while True:
        z = [rng.randrange(p) for _ in range(n)]
        if any(z):
            break
    u = [rng.randrange(p) for _ in range(n)]
    w = shifted_target(rows, inst["t"], u, p, radius)
    ts = add(inst["t"], w)
    Bs = sparsified_basis(rows, z, p)
    v, _ = oracle(embedding(Bs, ts, m))
    cs = extract(v, ts, m)
    if cs is None:
        return False
    c = sub(cs, w)
    return norm2(sub(inst["t"], c)) == inst["d2"]


def residue_vector(k, n, q):
    u = []
    for _ in range(n):
        u.append(k % q)
        k //= q
    return u


def det_prime(n, eps):
    N = tube_count_bound(n, eps)
    h = (1.0 + 1.0 / sqrt(eps)) / sqrt(2.0)
    return next_prime(max(2 * N * (n - 1) + 2 * n, int(ceil(h)) + 1)), N


def pencil_trial(inst, oracle, z, u, q, m):
    rows = inst["B"]
    w = combine(u, rows)
    ts = add(inst["t"], w)
    Bs = sparsified_basis(rows, z, q)
    v, _ = oracle(embedding(Bs, ts, m))
    cs = extract(v, ts, m)
    if cs is None:
        return False
    c = sub(cs, w)
    return norm2(sub(inst["t"], c)) == inst["d2"]


def bdd_deterministic(inst, oracle, eps=0.1, max_calls=None):
    oracle.reset()
    tracemalloc.start()
    t0 = time.perf_counter()
    n = inst["n"]
    d = sqrt(inst["d2"])
    q, N = det_prime(n, eps)
    m = max(1, int(round(sqrt(eps) * d)))
    success = False
    done = False
    for k in range(N * (q - 1) + 2):
        u = residue_vector(k, n, q)
        for z in pencil_family(n, q):
            if pencil_trial(inst, oracle, z, u, q, m):
                success = True
                done = True
                break
            if max_calls is not None and oracle.calls >= max_calls:
                done = True
                break
        if done:
            break
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"method": "DET", "success": success, "calls": oracle.calls, "valid_calls": oracle.valid_calls, "seconds": elapsed, "oracle_seconds": oracle.seconds, "peak_bytes": peak, "max_bits": oracle.max_bits, "prime": q}


def adaptive_levels(p0, cap):
    levels = []
    j = 0
    while True:
        p = next_prime(p0 * (1 << j))
        if p >= cap:
            break
        levels.append((p, int(ceil(2.0 * log(4.0) * p))))
        j += 1
    return levels


def adaptive_schedule(p0, cap):
    j = 0
    while True:
        p = next_prime(p0 * (1 << j))
        if p >= cap:
            return
        for _ in range(int(ceil(2.0 * log(4.0) * p))):
            yield p
        j += 1


def interleaved(inst, oracle, rng, fixed_p, m, radius, p0, budget):
    schedule = adaptive_schedule(p0, fixed_p)
    while oracle.calls < budget:
        if sparsified_trial(inst, oracle, rng, fixed_p, m, radius):
            return True
        q = next(schedule, None)
        if q is not None and sparsified_trial(inst, oracle, rng, q, m, radius):
            return True
    return False


def calibration_trial(inst, oracle, rng, p, m, radius):
    return sparsified_trial(inst, oracle, rng, p, m, radius)


def run_method(name, inst, oracle, gamma, seed, budget_factor=40, eps=0.1):
    rng = random.Random(seed)
    oracle.reset()
    tracemalloc.start()
    t0 = time.perf_counter()
    n = inst["n"]
    d = sqrt(inst["d2"])
    lam = sqrt(inst["lam2"])
    success = False
    p = 0
    if name == "KAN":
        success = single_call(inst, oracle, 1.0 / sqrt(n))
    elif name == "LM09":
        success = single_call(inst, oracle, 1.0)
    elif name == "FOLK":
        success = single_call(inst, oracle, folk_k(gamma))
    elif name == "BSW16":
        p = bsw16_prime(n, gamma)
        m = max(1, int(round(d / (n - 1))))
        radius = (n + 1) * lam / sqrt(2.0)
        for _ in range(budget_factor * p):
            if sparsified_trial(inst, oracle, rng, p, m, radius):
                success = True
                break
    elif name == "TSE":
        success = single_call(inst, oracle, folk_k(gamma))
        if not success:
            p = tse_prime(n, eps)
            m = max(1, int(round(sqrt(eps) * d)))
            radius = 0.0
            for _ in range(budget_factor * p):
                if sparsified_trial(inst, oracle, rng, p, m, radius):
                    success = True
                    break
    elif name == "ATSE":
        success = single_call(inst, oracle, folk_k(gamma))
        if not success:
            p = tse_prime(n, eps)
            m = max(1, int(round(sqrt(eps) * d)))
            p0 = next_prime(int(h_eps(eps)) + 1)
            success = interleaved(inst, oracle, rng, p, m, 0.0, p0, 1 + 2 * budget_factor * p)
    elif name == "TSEF":
        success = single_call(inst, oracle, folk_k(gamma))
        if not success:
            p = tse_prime_min(n, eps)
            m = max(1, int(round(sqrt(eps) * d)))
            for _ in range(budget_factor * p):
                if sparsified_trial(inst, oracle, rng, p, m, 0.0):
                    success = True
                    break
    elif name == "TSE-S":
        p = tse_prime(n, eps)
        m = max(1, int(round(sqrt(eps) * d)))
        for _ in range(budget_factor * p):
            if sparsified_trial(inst, oracle, rng, p, m, 0.0):
                success = True
                break
    elif name == "ATSE-S":
        p = tse_prime(n, eps)
        m = max(1, int(round(sqrt(eps) * d)))
        p0 = next_prime(int(h_eps(eps)) + 1)
        success = interleaved(inst, oracle, rng, p, m, 0.0, p0, 2 * budget_factor * p)
    elif name == "ABSW16":
        p = bsw16_prime(n, gamma)
        m = max(1, int(round(d / (n - 1))))
        radius = (n + 1) * lam / sqrt(2.0)
        success = interleaved(inst, oracle, rng, p, m, radius, 3, 2 * budget_factor * p)
    else:
        raise ValueError(name)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "method": name,
        "success": success,
        "calls": oracle.calls,
        "valid_calls": oracle.valid_calls,
        "seconds": elapsed,
        "oracle_seconds": oracle.seconds,
        "peak_bytes": peak,
        "max_bits": oracle.max_bits,
        "prime": p,
    }


def bdd_shift_deterministic(inst, oracle, eps=0.1, max_calls=None):
    oracle.reset()
    tracemalloc.start()
    t0 = time.perf_counter()
    n = inst["n"]
    d = sqrt(inst["d2"])
    N = tube_count_min(n, eps)
    q = next_prime(max(N * (n - 1) + 1, int(h_eps(eps)) + 2))
    m = max(1, int(round(sqrt(eps) * d)))
    rows = inst["B"]
    success = False
    done = False
    for a in range(N * (n - 1) + 1):
        z = moment(a, n, q)
        Bs = sparsified_basis(rows, z, q)
        for t in range(q):
            w = scale(t, rows[0])
            ts = add(inst["t"], w)
            v, _ = oracle(embedding(Bs, ts, m))
            cs = extract(v, ts, m)
            if cs is not None and norm2(sub(inst["t"], sub(cs, w))) == inst["d2"]:
                success = True
                done = True
                break
            if max_calls is not None and oracle.calls >= max_calls:
                done = True
                break
        if done:
            break
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"method": "DETS", "success": success, "calls": oracle.calls, "valid_calls": oracle.valid_calls, "seconds": elapsed, "oracle_seconds": oracle.seconds, "peak_bytes": peak, "max_bits": oracle.max_bits, "prime": q}


def zsvp_thin(rows, oracle, gamma2, a_bound, max_calls=None):
    n = len(rows)
    gamma = sqrt(float(gamma2))
    p = thin_isolation_prime(n, a_bound)
    if p <= 2 * gamma:
        p = next_prime(int(2 * gamma) + 1)
    best = None
    calls = 0
    for z in thin_pencil_family(n, p):
        Bs = sparsified_basis(rows, z, p)
        v, _ = oracle(Bs)
        calls += 1
        nv = norm2(v) if v is not None else 0
        if nv > 0 and (best is None or nv < best[0]):
            best = (nv, v)
        if best is not None and best[0] == 1:
            break
        if max_calls is not None and calls >= max_calls:
            break
    return best, calls, p


def zsvp_deterministic(rows, oracle, gamma2, a_bound, max_calls=None):
    n = len(rows)
    p = isolation_prime(n, a_bound)
    gamma = sqrt(float(gamma2))
    if p <= 2 * gamma:
        p = isolation_prime(n, a_bound + int(gamma) + 1)
    best = None
    calls = 0
    for z in pencil_family(n, p):
        Bs = sparsified_basis(rows, z, p)
        v, _ = oracle(Bs)
        calls += 1
        nv = norm2(v) if v is not None else 0
        if nv > 0 and (best is None or nv < best[0]):
            best = (nv, v)
        if best is not None and best[0] == 1:
            break
        if max_calls is not None and calls >= max_calls:
            break
    return best, calls, p


def zsvp_randomized(rows, oracle, gamma2, a_bound, seed, max_calls=None):
    rng = random.Random(seed)
    n = len(rows)
    p = isolation_prime(n, a_bound)
    best = None
    calls = 0
    while True:
        z = [rng.randrange(p) for _ in range(n)]
        if not any(z):
            continue
        Bs = sparsified_basis(rows, z, p)
        v, _ = oracle(Bs)
        calls += 1
        nv = norm2(v) if v is not None else 0
        if nv > 0 and (best is None or nv < best[0]):
            best = (nv, v)
        if best is not None and best[0] == 1:
            break
        if max_calls is not None and calls >= max_calls:
            break
    return best, calls, p


def zsvp_bgps(rows, oracle, a_bound, g_count, seed, max_calls=None):
    rng = random.Random(seed)
    n = len(rows)
    p = next_prime(int(ceil(50 * a_bound * log(100 * a_bound))))
    best = None
    calls = 0
    while True:
        z = [rng.randrange(p) for _ in range(n)]
        if any(z):
            Bs = sparsified_basis(rows, z, p)
        else:
            Bs = rows
        v, _ = oracle(Bs)
        calls += 1
        nv = norm2(v) if v is not None else 0
        if nv > 0 and (best is None or nv < best[0]):
            best = (nv, v)
        if best is not None and best[0] == 1:
            break
        if max_calls is not None and calls >= max_calls:
            break
    return best, calls, p
