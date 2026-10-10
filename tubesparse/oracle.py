import time
from fractions import Fraction
from math import ceil
from .lattice import shortest_list, lll_rows, parallel, bits_of, norm2


class PromiseUSVPOracle:
    def __init__(self, gamma2):
        self.gamma2 = Fraction(gamma2)
        self.k = 2 * ceil(float(self.gamma2) ** 0.5) + 3
        self.reset()

    def reset(self):
        self.calls = 0
        self.valid_calls = 0
        self.max_bits = 0
        self.max_dim = 0
        self.seconds = 0.0

    def __call__(self, rows):
        t0 = time.perf_counter()
        self.calls += 1
        b = bits_of(rows)
        if b > self.max_bits:
            self.max_bits = b
        if len(rows) > self.max_dim:
            self.max_dim = len(rows)
        red = lll_rows(rows)
        radius = int(self.gamma2 * norm2(red[0])) + 1
        sols, _ = shortest_list(red, self.k, radius2=radius)
        if not sols:
            sols = [(norm2(red[0]), red[0])]
        n1, v1 = sols[0]
        bound = self.gamma2 * n1
        valid = True
        for nn, v in sols[1:]:
            if not parallel(v, v1):
                if Fraction(nn) < bound:
                    valid = False
                break
        self.seconds += time.perf_counter() - t0
        if not valid:
            return None, False
        self.valid_calls += 1
        return v1, True


class HelpfulUSVPOracle(PromiseUSVPOracle):
    def __call__(self, rows):
        t0 = time.perf_counter()
        self.calls += 1
        b = bits_of(rows)
        if b > self.max_bits:
            self.max_bits = b
        red = lll_rows(rows)
        sols, _ = shortest_list(red, 1)
        v1 = sols[0][1] if sols else red[0]
        self.seconds += time.perf_counter() - t0
        return v1, True
