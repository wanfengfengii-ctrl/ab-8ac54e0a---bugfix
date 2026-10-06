"""Core exact computation: estimate and propagated variance.

Given inputs x_i with sensitivities c_i, intercept b and covariance matrix S:

    estimate = b + sum_i c_i * x_i
    variance = c^T S c = sum_i sum_j c_i * S[i][j] * c_j

Optional reference observations z = H x + e (e with covariance R) update the
result by exact linear conditioning on the joint model:

    estimate' = estimate + beta^T u
    variance' = variance - beta^T w
    M = H S H^T + R,  beta = H S c,
    M u = z - H x,   M w = beta

When M is invertible this is the usual formula with u = M^-1 (z - H x) and
w = M^-1 beta.  A singular M is accepted as long as the degenerate
observations are mutually consistent (z - H x lies in range(M)); only truly
contradictory degenerate observations make the conditional result undefined.

Everything is computed with :class:`fractions.Fraction`; no floating point
value is ever produced, so the budget verdict is exact.
"""
from __future__ import annotations

from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

from .matrix import Matrix, solve


def propagate(values: Sequence[Fraction],
              sensitivities: Sequence[Fraction],
              covariance: Matrix,
              intercept: Fraction) -> Tuple[Fraction, Fraction]:
    """Return ``(estimate, variance)`` computed with exact rational arithmetic."""
    estimate = intercept
    for sensitivity, value in zip(sensitivities, values):
        estimate += sensitivity * value

    n = len(values)
    variance = Fraction(0)
    for i in range(n):
        s_i = sensitivities[i]
        if not s_i:
            continue
        row = covariance[i]
        inner = Fraction(0)
        for j in range(n):
            s_j = sensitivities[j]
            if s_j:
                inner += row[j] * s_j
        variance += s_i * inner
    return estimate, variance


def condition(values: Sequence[Fraction],
              sensitivities: Sequence[Fraction],
              covariance: Matrix,
              intercept: Fraction,
              ref_values: Sequence[Fraction],
              ref_rows: Sequence[Sequence[Fraction]],
              ref_covariance: Matrix
              ) -> Optional[Tuple[Fraction, Fraction]]:
    """Return ``(estimate, variance)`` conditioned on reference observations.

    ``ref_rows`` is the m x n coefficient matrix H of the reference
    observations (row k maps the n inputs to observation k), ``ref_values``
    the m observed values z and ``ref_covariance`` the m x m reference noise
    covariance R.  Conditioning is exact::

        estimate' = estimate + beta^T u
        variance' = variance - beta^T w
        M = H S H^T + R,  beta = H S c,
        M u = z - H x,   M w = beta

    When ``M`` is invertible this is the usual Gaussian conditioning
    formula (``u = M^-1 (z - H x)``, ``w = M^-1 beta``).

    Degenerate (but consistent) references are accepted: ``M`` may be
    singular because of redundant noiseless observations (e.g. the same
    reference information recorded twice under different ids) or directions
    without prior variance.  Since ``M`` is positive semidefinite, the
    conditional estimate is unique exactly when the residual
    ``z - H x`` lies in the column space of ``M``; the observations are
    then mutually compatible.  Moreover ``beta = H S c`` always lies in
    ``range(H S H^T) subset range(M)`` (for symmetric PSD S,
    ``range(H S H^T) = range(H S^{1/2})`` and
    ``beta = (H S^{1/2})(S^{1/2} c)``), so the variance update is uniquely
    determined as well: both linear systems are consistent and their
    possibly non-unique solutions all yield the same scalar result
    ``beta^T u`` / ``beta^T w``.

    Returns ``None`` when the residual system is inconsistent: a singular
    ``M`` together with ``z - H x`` outside ``range(M)`` means the
    degenerate observations contradict each other and no unique
    conditional result exists.
    """
    estimate, variance = propagate(values, sensitivities, covariance, intercept)

    n = len(values)
    m = len(ref_values)

    # a = H S (m x n); only nonzero H entries contribute.
    a: Matrix = [[Fraction(0)] * n for _ in range(m)]
    for k in range(m):
        row_h = ref_rows[k]
        row_a = a[k]
        for i in range(n):
            h = row_h[i]
            if h:
                col_s = covariance[i]
                for j in range(n):
                    if col_s[j]:
                        row_a[j] += h * col_s[j]

    # M = H S H^T + R (m x m), beta = H S c = a c (m-vector).
    m_matrix: Matrix = [list(row) for row in ref_covariance]
    beta: List[Fraction] = [Fraction(0)] * m
    for k in range(m):
        row_a = a[k]
        row_m = m_matrix[k]
        for l in range(m):
            acc = row_m[l]
            row_h = ref_rows[l]
            for j in range(n):
                if row_a[j] and row_h[j]:
                    acc += row_a[j] * row_h[j]
            row_m[l] = acc
        for j in range(n):
            if row_a[j] and sensitivities[j]:
                beta[k] += row_a[j] * sensitivities[j]

    # residual r = z - H x (m-vector).
    residual: List[Fraction] = []
    for k in range(m):
        r = ref_values[k]
        row_h = ref_rows[k]
        for j in range(n):
            if row_h[j] and values[j]:
                r -= row_h[j] * values[j]
        residual.append(r)

    # M may be singular (redundant noiseless references).  The conditional
    # estimate exists uniquely iff M u = r is consistent; the degenerate
    # observations then agree instead of contradicting each other.  The
    # variance system M w = beta is always consistent because beta lies in
    # range(M), but it is checked defensively.
    u = solve(m_matrix, residual)
    if u is None:
        return None
    w = solve(m_matrix, beta)
    if w is None:
        return None

    # estimate' = estimate + beta^T u; variance' = variance - beta^T w.
    for k in range(m):
        if beta[k]:
            if u[k]:
                estimate += beta[k] * u[k]
            if w[k]:
                variance -= beta[k] * w[k]
    return estimate, variance
