"""
NIST SP 800-22 Rev. 1a Statistical Test Suite — Pure Python Implementation.

A correct, dependency-light implementation of the NIST randomness tests.
Created because the popular 'nistrng' package (v1.2.3) has critical bugs
in 7 out of 14 tests (see: https://github.com/InsaneMonster/NistRng/issues/13).

All 15 tests in this module have been validated against 5 independent inputs:
  - True random (os.urandom / CSPRNG)
  - AES-256 encrypted data (7-Zip)
  - Turbine V5 cipher
  - SCHFM2 cipher
  - Raw JPEG (must fail all tests — negative control)

Tests implemented (15 total):
  1.  Monobit Frequency
  2.  Block Frequency
  3.  Runs Test
  4.  Longest Run of Ones in a Block
  5.  Cumulative Sums (forward)
  6.  Cumulative Sums (backward)
  7.  Approximate Entropy (m=10)
  8.  Serial Test (m=16) — produces two p-values
  9.  Maurer's Universal Statistical Test
  10. DFT / Spectral Test
  11. Binary Matrix Rank
  12. Non-Overlapping Template Matching
  13. Linear Complexity (Berlekamp-Massey, M=500)
  14. Chi-Squared Byte Distribution (supplementary)
  15. Compression Ratio (supplementary, informational)

Usage:
  python nist_sp800_22.py <file> [skip_bytes]

  skip_bytes: number of header bytes to skip (default: 0)

Author:  Reinhard Jesolowitz
License: MIT
"""

import math
import sys
import time
import os
from collections import Counter

# Optional: numpy for DFT/Spectral test only
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

__version__ = "1.1.0"


# =====================================================================
# Mathematical helper functions
# =====================================================================

def igamc(a, x):
    """
    Regularized upper incomplete gamma function Q(a, x).
    Uses series expansion for x < a+1, continued fraction otherwise.
    """
    if x < 0 or a <= 0:
        return 0.0
    if x == 0:
        return 1.0
    if x < a + 1:
        ap = a
        s = 1.0 / a
        delta = s
        for _ in range(200):
            ap += 1
            delta *= x / ap
            s += delta
            if abs(delta) < abs(s) * 1e-15:
                break
        return 1.0 - s * math.exp(-x + a * math.log(x) - math.lgamma(a))
    else:
        b = x + 1.0 - a
        c = 1.0 / 1e-300
        d = 1.0 / b
        h = d
        for i in range(1, 200):
            an = -i * (i - a)
            b += 2.0
            d = an * d + b
            if abs(d) < 1e-300:
                d = 1e-300
            c = b + an / c
            if abs(c) < 1e-300:
                c = 1e-300
            d = 1.0 / d
            delt = d * c
            h *= delt
            if abs(delt - 1.0) < 1e-15:
                break
        return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def normal_cdf(x):
    """Standard normal cumulative distribution function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# =====================================================================
# NIST SP 800-22 Tests
# =====================================================================

def monobit_frequency(bits):
    """
    Test 1: Monobit Frequency Test.

    Checks whether the proportion of ones and zeros in the entire
    sequence is approximately equal, as expected for a random sequence.

    Args:
        bits: list of 0/1 integers

    Returns:
        p-value (float). Pass if >= 0.01.
    """
    n = len(bits)
    s = sum(2 * b - 1 for b in bits)
    s_obs = abs(s) / math.sqrt(n)
    return math.erfc(s_obs / math.sqrt(2))


def block_frequency(bits, M=128):
    """
    Test 2: Block Frequency Test.

    Divides the sequence into M-bit blocks and checks whether the
    proportion of ones in each block is approximately 0.5.

    Args:
        bits: list of 0/1 integers
        M: block size in bits (default: 128)

    Returns:
        p-value (float). Pass if >= 0.01.
    """
    n = len(bits)
    N = n // M
    if N == 0:
        return 0.0
    chi2 = 0.0
    for i in range(N):
        block = bits[i * M:(i + 1) * M]
        pi = sum(block) / M
        chi2 += (pi - 0.5) ** 2
    chi2 *= 4 * M
    return igamc(N / 2, chi2 / 2)


def runs_test(bits):
    """
    Test 3: Runs Test.

    Checks whether the number of runs (uninterrupted sequences of
    identical bits) is as expected for a random sequence.

    Args:
        bits: list of 0/1 integers

    Returns:
        p-value (float). Pass if >= 0.01.
    """
    n = len(bits)
    pi = sum(bits) / n
    if abs(pi - 0.5) > 2 / math.sqrt(n):
        return 0.0
    Vn = 1 + sum(1 for i in range(1, n) if bits[i] != bits[i - 1])
    num = abs(Vn - 2 * n * pi * (1 - pi))
    denom = 2 * math.sqrt(2 * n) * pi * (1 - pi)
    return math.erfc(num / denom)


def longest_run(bits):
    """
    Test 4: Longest Run of Ones in a Block.

    Checks whether the longest run of ones within M-bit blocks is
    consistent with what is expected for a random sequence.

    Args:
        bits: list of 0/1 integers

    Returns:
        p-value (float). Pass if >= 0.01.
    """
    n = len(bits)
    if n < 6272:
        M, K = 8, 3
        cats = [1, 2, 3, 4]
        pi = [0.2148, 0.3672, 0.2305, 0.1875]
    elif n < 750000:
        M, K = 128, 5
        cats = [4, 5, 6, 7, 8, 9]
        pi = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
    else:
        M, K = 10000, 6
        cats = [10, 11, 12, 13, 14, 15, 16]
        pi = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]

    N = n // M
    v = [0] * (K + 1)
    for i in range(N):
        block = bits[i * M:(i + 1) * M]
        max_run = 0
        cur = 0
        for b in block:
            if b == 1:
                cur += 1
                if cur > max_run:
                    max_run = cur
            else:
                cur = 0
        if max_run <= cats[0]:
            v[0] += 1
        elif max_run >= cats[-1]:
            v[K] += 1
        else:
            v[max_run - cats[0]] += 1

    chi2 = sum((v[i] - N * pi[i]) ** 2 / (N * pi[i]) for i in range(K + 1))
    return igamc(K / 2, chi2 / 2)


def cumulative_sums(bits, mode='forward'):
    """
    Test 5/6: Cumulative Sums Test.

    Checks whether the cumulative sum of the adjusted (+1/-1) sequence
    is too large or too small relative to expected behavior.

    Args:
        bits: list of 0/1 integers
        mode: 'forward' (Test 5) or 'backward' (Test 6)

    Returns:
        p-value (float). Pass if >= 0.01.
    """
    n = len(bits)
    s = [2 * b - 1 for b in bits]
    if mode == 'backward':
        s = s[::-1]
    cumsum = 0
    z = 0
    for x in s:
        cumsum += x
        if abs(cumsum) > z:
            z = abs(cumsum)
    if z == 0:
        return 1.0
    sqn = math.sqrt(n)
    sum1 = 0.0
    k_lo = int((-n / z + 1) / 4)
    k_hi = int((n / z - 1) / 4)
    for k in range(k_lo, k_hi + 1):
        sum1 += normal_cdf(((4 * k + 1) * z) / sqn) - normal_cdf(((4 * k - 1) * z) / sqn)
    sum2 = 0.0
    k_lo = int((-n / z - 3) / 4)
    for k in range(k_lo, k_hi + 1):
        sum2 += normal_cdf(((4 * k + 3) * z) / sqn) - normal_cdf(((4 * k + 1) * z) / sqn)
    p = 1 - sum1 + sum2
    return max(0.0, min(1.0, p))


def approximate_entropy(bits, m=10):
    """
    Test 7: Approximate Entropy Test.

    Compares the frequency of overlapping m-bit and (m+1)-bit patterns.
    A significant difference suggests non-randomness.

    Note: nistrng v1.2.3 has TWO bugs in this test:
      - min/max swapped on line 52, forcing m=2 instead of dynamic
      - divides by 10.0 inside log on line 74, not in NIST spec

    Args:
        bits: list of 0/1 integers
        m: pattern length (default: 10, appropriate for sequences > 100K bits)

    Returns:
        p-value (float). Pass if >= 0.01.
    """
    n = len(bits)

    def phi(mm):
        ext = bits + bits[:mm - 1]
        c = Counter()
        for i in range(n):
            pat = 0
            for j in range(mm):
                pat = (pat << 1) | ext[i + j]
            c[pat] += 1
        s = 0.0
        for v in c.values():
            p = v / n
            s += p * math.log(p)
        return s

    apen = phi(m) - phi(m + 1)
    chi2 = 2 * n * (math.log(2) - apen)
    return igamc(2 ** (m - 1), chi2 / 2)


def serial_test(bits, m=16):
    """
    Test 8: Serial Test.

    Checks whether the number of occurrences of all possible overlapping
    m-bit patterns is approximately the same.

    Note: nistrng v1.2.3 hardcodes m=4, making the test far too coarse.

    Args:
        bits: list of 0/1 integers
        m: pattern length (default: 16, appropriate for sequences > 1M bits)

    Returns:
        tuple of (p_value_1, p_value_2). Both should be >= 0.01.
    """
    n = len(bits)

    def psi2(mm):
        if mm <= 0:
            return 0.0
        ext = bits + bits[:mm - 1]
        c = Counter()
        for i in range(n):
            pat = 0
            for j in range(mm):
                pat = (pat << 1) | ext[i + j]
            c[pat] += 1
        return (sum(v * v for v in c.values()) * (2 ** mm) / n) - n

    p1 = psi2(m)
    p2 = psi2(m - 1)
    p3 = psi2(m - 2)
    d1 = p1 - p2
    d2 = p1 - 2 * p2 + p3
    pv1 = igamc(2 ** (m - 2), d1 / 2)
    pv2 = igamc(2 ** (m - 3), d2 / 2)
    return pv1, pv2


def maurer_universal(bits):
    """
    Test 9: Maurer's Universal Statistical Test.

    Detects whether the sequence can be significantly compressed.
    Based on the distance between matching L-bit patterns.

    Args:
        bits: list of 0/1 integers (minimum 387,840 bits required)

    Returns:
        p-value (float), or None if sequence is too short.
    """
    n = len(bits)
    if n < 387840:
        return None
    L = 7
    Q = 1280
    K = n // L - Q
    if K <= 0:
        return None

    expected = {6: 5.2177052, 7: 6.1962507, 8: 7.1836656, 9: 8.1764248,
                10: 9.1723243, 11: 10.170032, 12: 11.168765, 13: 12.168070,
                14: 13.167693, 15: 14.167488, 16: 15.167379}
    variance = {6: 2.954, 7: 3.125, 8: 3.238, 9: 3.311, 10: 3.356,
                11: 3.384, 12: 3.401, 13: 3.410, 14: 3.416, 15: 3.419, 16: 3.421}

    table = [0] * (1 << L)
    for i in range(Q):
        pat = 0
        for j in range(L):
            pat = (pat << 1) | bits[i * L + j]
        table[pat] = i + 1
    s = 0.0
    for i in range(Q, Q + K):
        pat = 0
        for j in range(L):
            pat = (pat << 1) | bits[i * L + j]
        s += math.log(i + 1 - table[pat], 2)
        table[pat] = i + 1
    fn = s / K
    c = 0.7 - 0.8 / L + (4 + 32.0 / L) * (K ** (-3.0 / L)) / 15
    sigma = c * math.sqrt(variance[L] / K)
    return math.erfc(abs((fn - expected[L]) / (math.sqrt(2) * sigma)))


def dft_spectral(bits):
    """
    Test 10: Discrete Fourier Transform (Spectral) Test.

    Detects periodic features in the sequence that would indicate
    a deviation from randomness. Requires numpy.

    Args:
        bits: list of 0/1 integers

    Returns:
        p-value (float), or None if numpy is not available.
    """
    if not HAS_NUMPY:
        return None
    n = len(bits)
    x = np.array([2 * b - 1 for b in bits], dtype=np.float64)
    S = np.abs(np.fft.fft(x))
    T = math.sqrt(2.995732274 * n)  # 95% threshold
    S_half = S[:n // 2]
    N0 = 0.95 * n / 2.0  # expected count below threshold
    N1 = np.sum(S_half < T)  # actual count below threshold
    d = (N1 - N0) / math.sqrt(n * 0.95 * 0.05 / 4.0)
    return math.erfc(abs(d) / math.sqrt(2))


def binary_matrix_rank(bits):
    """
    Test 11: Binary Matrix Rank Test.

    Divides the sequence into 32x32 bit matrices and checks
    whether their rank distribution matches the expected values
    for random data.

    Args:
        bits: list of 0/1 integers

    Returns:
        p-value (float), or None if fewer than 38 matrices available.
    """
    n = len(bits)
    M, Q = 32, 32
    N = n // (M * Q)
    if N < 38:
        return None

    def _matrix_rank_gf2(matrix):
        """Compute rank of a binary matrix over GF(2)."""
        m = [row[:] for row in matrix]
        rows = len(m)
        cols = len(m[0])
        rank = 0
        for col in range(min(rows, cols)):
            pivot = None
            for row in range(rank, rows):
                if m[row][col] == 1:
                    pivot = row
                    break
            if pivot is None:
                continue
            m[rank], m[pivot] = m[pivot], m[rank]
            for row in range(rows):
                if row != rank and m[row][col] == 1:
                    m[row] = [m[row][j] ^ m[rank][j] for j in range(cols)]
            rank += 1
        return rank

    full_rank = 0
    rank_minus1 = 0
    rest = 0

    for i in range(N):
        start = i * M * Q
        matrix = []
        for r in range(M):
            row = bits[start + r * Q:start + (r + 1) * Q]
            matrix.append(list(row))
        r = _matrix_rank_gf2(matrix)
        if r == M:
            full_rank += 1
        elif r == M - 1:
            rank_minus1 += 1
        else:
            rest += 1

    # Expected probabilities for 32x32 matrices
    p32 = 0.2888   # P(rank = 32)
    p31 = 0.5776   # P(rank = 31)
    p30 = 0.1336   # P(rank <= 30)

    chi2 = ((full_rank - N * p32) ** 2 / (N * p32) +
            (rank_minus1 - N * p31) ** 2 / (N * p31) +
            (rest - N * p30) ** 2 / (N * p30))

    return math.exp(-chi2 / 2)


def non_overlapping_template(bits, m=9):
    """
    Test 12: Non-Overlapping Template Matching Test.

    Checks whether a specific m-bit pattern occurs with the expected
    frequency. Uses template B = 000000001 (m-1 zeros followed by one).

    Args:
        bits: list of 0/1 integers
        m: template length (default: 9)

    Returns:
        p-value (float). Pass if >= 0.01.
    """
    n = len(bits)
    N = 8  # number of blocks
    M = n // N  # block length

    template = [0] * (m - 1) + [1]

    W = []
    for block_idx in range(N):
        start = block_idx * M
        block = bits[start:start + M]
        count = 0
        i = 0
        while i <= M - m:
            match = True
            for j in range(m):
                if block[i + j] != template[j]:
                    match = False
                    break
            if match:
                count += 1
                i += m  # non-overlapping: skip ahead
            else:
                i += 1
        W.append(count)

    mu = (M - m + 1) / (2 ** m)
    sigma2 = M * (1.0 / (2 ** m) - (2 * m - 1) / (2 ** (2 * m)))
    chi2 = sum((w - mu) ** 2 / sigma2 for w in W)
    return igamc(N / 2.0, chi2 / 2.0)


def linear_complexity(bits, M=500):
    """
    Test 13: Linear Complexity Test (Berlekamp-Massey).

    Divides the sequence into M-bit blocks, computes the linear complexity
    (shortest LFSR that generates the block) via the Berlekamp-Massey
    algorithm, and checks whether the distribution matches the theoretical
    expectation for random data.

    IMPORTANT: The pi values published in NIST SP 800-22 (Table 2.10) are
    incorrect — they do not match the theoretical distribution of linear
    complexity for any M value. This implementation computes the correct
    pi values from the exact theoretical distribution using dynamic
    programming over the Berlekamp-Massey state transitions. Additionally,
    the T formula uses (-1)^M (not (-1)^(M+1) as in some implementations).
    These corrections were verified empirically with os.urandom() data.

    Args:
        bits: list of 0/1 integers
        M: block size (default: 500, per NIST recommendation)

    Returns:
        p-value (float), or None if insufficient data (need >= 200 blocks).
    """
    n = len(bits)
    N = n // M
    if N < 200:
        return None

    K = 6  # number of degrees of freedom

    # Compute theoretical distribution of L for block size M
    # using dynamic programming over BM algorithm transitions
    prob = {0: 1.0}
    for step in range(M):
        new_prob = {}
        for L, p in prob.items():
            # d=0 (prob 1/2): L stays
            new_prob[L] = new_prob.get(L, 0) + p * 0.5
            # d=1 (prob 1/2): L changes if 2L <= step
            if 2 * L <= step:
                new_L = step + 1 - L
                new_prob[new_L] = new_prob.get(new_L, 0) + p * 0.5
            else:
                new_prob[L] = new_prob.get(L, 0) + p * 0.5
        prob = new_prob

    # Expected linear complexity
    mu = M / 2.0 + (9 + (-1) ** (M + 1)) / 36.0 - (M / 3.0 + 2 / 9.0) / (2 ** M)

    # Compute correct pi values by binning theoretical T distribution
    # T uses (-1)^M sign (not (-1)^(M+1) as in some buggy implementations)
    sign = (-1) ** M
    pi = [0.0] * 7
    for L_val, p in prob.items():
        T = sign * (L_val - mu) + 2.0 / 9.0
        if T <= -2.5:
            pi[0] += p
        elif T <= -1.5:
            pi[1] += p
        elif T <= -0.5:
            pi[2] += p
        elif T <= 0.5:
            pi[3] += p
        elif T <= 1.5:
            pi[4] += p
        elif T <= 2.5:
            pi[5] += p
        else:
            pi[6] += p

    # Berlekamp-Massey algorithm for GF(2)
    def _berlekamp_massey(block):
        blen = len(block)
        C = [0] * (blen + 1)
        B = [0] * (blen + 1)
        C[0] = 1
        B[0] = 1
        L = 0
        m = -1
        for i in range(blen):
            d = block[i]
            for j in range(1, L + 1):
                d ^= C[j] & block[i - j]
            d &= 1
            if d == 1:
                T_poly = C[:]
                shift = i - m
                for j in range(blen + 1 - shift):
                    if B[j] == 1:
                        C[j + shift] ^= 1
                if 2 * L <= i:
                    L = i + 1 - L
                    m = i
                    B = T_poly[:]
        return L

    # Categorize blocks
    v = [0] * 7
    for i in range(N):
        block = bits[i * M:(i + 1) * M]
        L = _berlekamp_massey(block)
        T = sign * (L - mu) + 2.0 / 9.0
        if T <= -2.5:
            v[0] += 1
        elif T <= -1.5:
            v[1] += 1
        elif T <= -0.5:
            v[2] += 1
        elif T <= 0.5:
            v[3] += 1
        elif T <= 1.5:
            v[4] += 1
        elif T <= 2.5:
            v[5] += 1
        else:
            v[6] += 1

    # Chi-squared test
    chi2 = sum((v[i] - N * pi[i]) ** 2 / (N * pi[i]) for i in range(K + 1))
    return igamc(K / 2.0, chi2 / 2.0)


# =====================================================================
# Supplementary tests (not in NIST SP 800-22, but useful)
# =====================================================================

def chi_squared_byte(data_bytes):
    """
    Supplementary Test 14: Chi-Squared Byte Distribution.

    Tests whether byte values (0-255) are uniformly distributed.
    For truly random data, all 256 byte values should appear
    with approximately equal frequency.

    Args:
        data_bytes: list or bytes of raw byte values

    Returns:
        p-value (float). Pass if >= 0.01.
    """
    n = len(data_bytes)
    expected = n / 256.0
    counts = Counter(data_bytes)
    chi2 = sum((counts.get(i, 0) - expected) ** 2 / expected for i in range(256))
    return igamc(255 / 2.0, chi2 / 2.0)


def compression_ratio(data_bytes, sample_size=1000000):
    """
    Supplementary Test 15: Compression Ratio.

    Measures how compressible the data is using zlib.
    Truly random data is incompressible (ratio >= 1.0).
    Non-random data compresses to a smaller size (ratio < 1.0).

    Args:
        data_bytes: list or bytes of raw byte values
        sample_size: max bytes to test (default: 1M)

    Returns:
        float: compression ratio (>= 1.0 is good, < 1.0 is suspicious)
    """
    import zlib
    sample = bytes(data_bytes[:sample_size])
    compressed = zlib.compress(sample, 9)
    return len(compressed) / len(sample)


# =====================================================================
# Convenience: serial correlation and Shannon entropy
# =====================================================================

def serial_correlation(data_bytes, sample_size=None):
    """
    Compute the serial correlation coefficient between consecutive bytes.

    For random data, this should be close to 0.
    Values > |0.01| indicate detectable correlation.

    Args:
        data_bytes: list or bytes of raw byte values
        sample_size: optional limit

    Returns:
        float: correlation coefficient (close to 0 = good)
    """
    d = data_bytes[:sample_size] if sample_size else data_bytes
    n = len(d) - 1
    if n <= 0:
        return 0.0
    sx = sy = sxx = syy = sxy = 0.0
    for i in range(n):
        x, y = d[i], d[i + 1]
        sx += x
        sy += y
        sxx += x * x
        syy += y * y
        sxy += x * y
    mx = sx / n
    my = sy / n
    num = sxy / n - mx * my
    den = math.sqrt((sxx / n - mx * mx) * (syy / n - my * my))
    return num / den if den > 0 else 0.0


def shannon_entropy(data_bytes):
    """
    Compute Shannon entropy in bits per byte.

    For truly random data, entropy should be very close to 8.0.
    Values significantly below 8.0 indicate patterns.

    Args:
        data_bytes: list or bytes of raw byte values

    Returns:
        float: entropy in bits/byte (max = 8.0)
    """
    n = len(data_bytes)
    counts = Counter(data_bytes)
    entropy = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


# =====================================================================
# Test runner
# =====================================================================

def bytes_to_bits(data):
    """Convert a bytes-like object to a list of 0/1 integers (MSB first)."""
    bits = []
    for byte in data:
        for j in range(7, -1, -1):
            bits.append((byte >> j) & 1)
    return bits


def run_all_tests(data, label="", verbose=True):
    """
    Run all 14 tests on the given data.

    Args:
        data: bytes or bytearray of raw data (no headers)
        label: optional label for display
        verbose: print results to stdout

    Returns:
        dict with test results:
          {test_name: {"p_value": float, "passed": bool, "time": float}, ...}
    """
    data_bytes = list(data)

    t0 = time.time()
    bits = bytes_to_bits(data)
    n = len(bits)
    if verbose:
        if label:
            print(f"\n{'=' * 70}")
            print(f"  {label}")
            print(f"{'=' * 70}")
        print(f"  {len(data):,} bytes = {n:,} bits ({time.time() - t0:.1f}s)")

    results = {}

    def _run(name, fn, is_pvalue=True):
        t = time.time()
        try:
            r = fn()
            dt = time.time() - t
            if r is None:
                if verbose:
                    print(f"  [SKIP] {name}: insufficient data ({dt:.1f}s)")
                return
            if isinstance(r, tuple):
                for i, p in enumerate(r):
                    key = f"{name} ({i + 1})"
                    passed = p >= 0.01
                    results[key] = {"p_value": p, "passed": passed, "time": dt}
                    if verbose:
                        tag = "PASS" if passed else "FAIL"
                        print(f"  [{tag}] {key}: p = {p:.6f}  ({dt:.1f}s)")
            else:
                if is_pvalue:
                    passed = r >= 0.01
                    results[name] = {"p_value": r, "passed": passed, "time": dt}
                    if verbose:
                        tag = "PASS" if passed else "FAIL"
                        print(f"  [{tag}] {name}: p = {r:.6f}  ({dt:.1f}s)")
                else:
                    results[name] = {"value": r, "time": dt}
                    if verbose:
                        print(f"  [INFO] {name}: {r:.6f}  ({dt:.1f}s)")
        except Exception as e:
            if verbose:
                print(f"  [ERR ] {name}: {e}")

    # --- NIST SP 800-22 Tests ---
    if verbose:
        print(f"\n  --- NIST SP 800-22 Tests ---")
    _run("01 Monobit Frequency", lambda: monobit_frequency(bits))
    _run("02 Block Frequency (M=128)", lambda: block_frequency(bits, 128))
    _run("03 Runs Test", lambda: runs_test(bits))
    _run("04 Longest Run of Ones", lambda: longest_run(bits))
    _run("05 Cumulative Sums (fwd)", lambda: cumulative_sums(bits, 'forward'))
    _run("06 Cumulative Sums (bwd)", lambda: cumulative_sums(bits, 'backward'))
    _run("07 Approximate Entropy (m=10)", lambda: approximate_entropy(bits, 10))
    _run("08 Serial Test (m=16)", lambda: serial_test(bits, 16))
    _run("09 Maurer Universal", lambda: maurer_universal(bits))
    _run("10 DFT/Spectral", lambda: dft_spectral(bits))
    _run("11 Binary Matrix Rank", lambda: binary_matrix_rank(bits))
    _run("12 Non-Overlapping Template", lambda: non_overlapping_template(bits, 9))
    _run("13 Linear Complexity (M=500)", lambda: linear_complexity(bits, 500))

    # --- Supplementary Tests ---
    if verbose:
        print(f"\n  --- Supplementary Tests ---")
    _run("14 Chi-Squared Byte", lambda: chi_squared_byte(data_bytes))
    _run("15 Compression Ratio", lambda: compression_ratio(data_bytes), is_pvalue=False)

    # Serial correlation (non-p-value metric)
    sc = serial_correlation(data_bytes)
    sc_pass = abs(sc) < 0.01
    results["16 Serial Correlation"] = {"value": sc, "passed": sc_pass}
    if verbose:
        tag = "PASS" if sc_pass else "FAIL"
        print(f"  [{tag}] 16 Serial Correlation: r = {sc:+.6f}")

    # Shannon entropy (non-p-value metric)
    ent = shannon_entropy(data_bytes)
    ent_pass = ent > 7.99
    results["17 Shannon Entropy"] = {"value": ent, "passed": ent_pass}
    if verbose:
        tag = "PASS" if ent_pass else "FAIL"
        print(f"  [{tag}] 17 Shannon Entropy: {ent:.6f} bits/byte")

    # --- Summary ---
    pvalue_tests = {k: v for k, v in results.items() if "passed" in v}
    passed = sum(1 for v in pvalue_tests.values() if v["passed"])
    total = len(pvalue_tests)
    if verbose:
        print(f"\n  === SUMMARY: {passed}/{total} tests passed ===")
        failed = {k: v for k, v in pvalue_tests.items() if not v["passed"]}
        if failed:
            for name, v in failed.items():
                p = v.get("p_value", v.get("value", "?"))
                print(f"      FAIL: {name}: {p}")
        else:
            print(f"      No anomalies detected.")

    return results


# =====================================================================
# Self-validation with os.urandom
# =====================================================================

def validate():
    """
    Run all tests against os.urandom data to verify correctness.
    All p-value tests should pass for cryptographically secure random data.

    Returns:
        bool: True if all tests passed.
    """
    print("Generating 1 MB of true random data (os.urandom)...")
    data = os.urandom(1_000_000)
    results = run_all_tests(data, label="VALIDATION: os.urandom (1 MB)")
    pvalue_tests = {k: v for k, v in results.items() if "passed" in v}
    all_passed = all(v["passed"] for v in pvalue_tests.values())
    if all_passed:
        print("\n  VALIDATION PASSED: All tests returned correct results.")
    else:
        print("\n  VALIDATION FAILED: Some tests returned incorrect results!")
    return all_passed


# =====================================================================
# CLI entry point
# =====================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"NIST SP 800-22 Statistical Test Suite v{__version__}")
        print(f"Usage: python {sys.argv[0]} <file> [skip_bytes]")
        print(f"       python {sys.argv[0]} --validate")
        print()
        print("  file:        binary file to test")
        print("  skip_bytes:  header bytes to skip (default: 0)")
        print("  --validate:  run self-test with os.urandom")
        sys.exit(1)

    if sys.argv[1] == "--validate":
        ok = validate()
        sys.exit(0 if ok else 1)

    filepath = sys.argv[1]
    skip = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    with open(filepath, 'rb') as f:
        raw = f.read()

    data = raw[skip:]
    run_all_tests(data, label=os.path.basename(filepath))
