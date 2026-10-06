import unittest
from fractions import Fraction as F

from app.matrix import check_symmetric, invert, psd_failure_index, solve


def m(rows):
    return [[F(v) for v in row] for row in rows]


class CheckSymmetricTests(unittest.TestCase):
    def test_symmetric(self):
        self.assertIsNone(check_symmetric(m([[1, 2], [2, 3]])))
        self.assertIsNone(check_symmetric(m([[0]])))
        self.assertIsNone(check_symmetric(
            [[F(1, 2), F(1, 3)], [F(1, 3), F(1, 4)]]))

    def test_first_mismatch_reported(self):
        self.assertEqual(check_symmetric(m([[1, 5], [2, 3]])), (0, 1))
        self.assertEqual(
            check_symmetric(m([[1, 2, 3], [2, 1, 4], [3, 5, 1]])), (1, 2))


class PsdFailureIndexTests(unittest.TestCase):
    def assert_psd(self, rows):
        self.assertIsNone(psd_failure_index(m(rows)), msg=repr(rows))

    def assert_not_psd(self, rows, pivot):
        self.assertEqual(psd_failure_index(m(rows)), pivot, msg=repr(rows))

    def test_positive_definite(self):
        self.assert_psd([[1]])
        self.assert_psd([[2, 1], [1, 2]])
        self.assert_psd([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
        self.assert_psd([[F(1, 2), F(1, 4)], [F(1, 4), F(1, 2)]])

    def test_positive_semidefinite_singular(self):
        self.assert_psd([[0]])
        self.assert_psd([[0, 0], [0, 0]])
        self.assert_psd([[0, 0], [0, 5]])
        self.assert_psd([[1, 2], [2, 4]])               # rank one
        self.assert_psd([[2, 0, 0], [0, 0, 0], [0, 0, 3]])
        self.assert_psd([[F(1, 2), F(1, 4)], [F(1, 4), F(1, 8)]])  # rank one
        self.assert_psd([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    def test_indefinite(self):
        self.assert_not_psd([[-1]], 0)
        self.assert_not_psd([[1, 2], [2, 1]], 1)
        self.assert_not_psd([[0, 1], [1, 0]], 0)   # zero pivot, nonzero row
        self.assert_not_psd([[0, 0], [0, -1]], 1)
        self.assert_not_psd([[1, 0, 0], [0, -2, 0], [0, 0, 1]], 1)

    def test_exact_fraction_pivot(self):
        # Second pivot is exactly 1/16 - (1/4)^2/(1/2) = -1/16 < 0.
        self.assert_not_psd(
            [[F(1, 2), F(1, 4)], [F(1, 4), F(1, 16)]], 1)


class InvertTests(unittest.TestCase):
    def assert_inverse(self, rows, expected):
        result = invert(m(rows))
        self.assertEqual(result, [[F(v) for v in row] for row in expected])

    def test_identity(self):
        self.assert_inverse([[1, 0], [0, 1]], [[1, 0], [0, 1]])

    def test_diagonal(self):
        self.assert_inverse([[2, 0], [0, F(1, 3)]], [[F(1, 2), 0], [0, 3]])

    def test_general_exact(self):
        # [[2, 1], [1, 2]]^-1 = 1/3 * [[2, -1], [-1, 2]]
        self.assert_inverse(
            [[2, 1], [1, 2]],
            [[F(2, 3), F(-1, 3)], [F(-1, 3), F(2, 3)]])

    def test_requires_pivoting(self):
        self.assert_inverse([[0, 1], [1, 0]], [[0, 1], [1, 0]])

    def test_product_is_identity(self):
        rows = [[F(1, 2), F(1, 4)], [F(1, 4), F(1, 2)]]
        inv = invert(rows)
        n = len(rows)
        for i in range(n):
            for j in range(n):
                entry = sum(rows[i][k] * inv[k][j] for k in range(n))
                self.assertEqual(entry, F(1) if i == j else F(0))

    def test_singular_returns_none(self):
        self.assertIsNone(invert(m([[0]])))
        self.assertIsNone(invert(m([[1, 2], [2, 4]])))
        self.assertIsNone(invert(m([[1, 1], [1, 1]])))
        self.assertIsNone(invert(m([[0, 0], [0, 5]])))


class SolveTests(unittest.TestCase):
    def assert_solves(self, rows, rhs, expected):
        result = solve(m(rows), [F(v) for v in rhs])
        self.assertIsNotNone(result, msg=repr((rows, rhs)))
        n = len(rows)
        for i in range(n):
            self.assertEqual(sum(rows[i][j] * result[j] for j in range(n)),
                             F(rhs[i]))
        self.assertEqual(result, [F(v) for v in expected])

    def test_invertible_systems(self):
        self.assert_solves([[1, 0], [0, 1]], [2, 3], [2, 3])
        self.assert_solves([[2, 1], [1, 2]], [1, 1], [F(1, 3), F(1, 3)])
        # Requires row pivoting.
        self.assert_solves([[0, 1], [1, 0]], [7, 8], [8, 7])

    def test_zero_rhs_with_singular_matrix(self):
        # 1x1 zero matrix with zero rhs is consistent (x = 0).
        self.assert_solves([[0]], [0], [0])
        # Rank-one 2x2 with an rhs in its column space.
        self.assert_solves([[1, 1], [1, 1]], [2, 2], [2, 0])

    def test_consistent_free_variable_particular_solution(self):
        # Rank one: x1 + x2 = 3; free variable x2 = 0 -> x1 = 3.
        self.assert_solves([[1, 1], [0, 0]], [3, 0], [3, 0])

    def test_inconsistent_systems_return_none(self):
        self.assertIsNone(solve(m([[0]]), [F(1)]))
        self.assertIsNone(solve(m([[1, 1], [1, 1]]), [F(1), F(2)]))
        self.assertIsNone(solve(m([[0, 0], [0, 0]]), [F(0), F(1)]))

    def test_exact_fractions(self):
        result = solve(m([[F(1, 2)]]), [F(1, 3)])
        self.assertEqual(result, [F(2, 3)])
        self.assertIsInstance(result[0], F)


if __name__ == "__main__":
    unittest.main()
