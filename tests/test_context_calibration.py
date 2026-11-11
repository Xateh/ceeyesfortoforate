"""Training exclusions and frozen calibration decision boundaries."""
import unittest

import numpy as np

from scripts.reproduce_context_results import fit, predict


class ContextCalibrationTest(unittest.TestCase):
    def test_absolute_f1_weights_change_the_intercept(self):
        # With identical features only the weighted class mass determines odds.
        train = dict(x=np.zeros((4, 2)), delta=np.array([1., -.1, -.1, -.1]))
        frozen = fit(train, [0, 1])
        self.assertAlmostEqual(frozen['intercept'], np.log(1. / .3), places=3)
        self.assertEqual(predict(frozen, train['x']).tolist(), [True] * 4)

    def test_equal_f1_pairs_do_not_affect_fit_or_scaling(self):
        train = dict(x=np.array([[1., -2.], [3., 2.], [2., 1.], [4., -1.]]),
                     delta=np.array([-.5, 1., .25, -1.]))
        with_tie = dict(x=np.vstack([train['x'], [1e9, -1e9]]),
                        delta=np.append(train['delta'], 0.))
        for columns in ([0], [0, 1]):
            original, extended = fit(train, columns), fit(with_tie, columns)
            self.assertEqual(original, extended)
            self.assertEqual(original['informative_pairs'], 4)

    def test_frozen_signed_correction_and_strict_zero_threshold(self):
        frozen = dict(mean=[2., 10.], scale=[2., 5.], coef=[-1., 2.], intercept=0.)
        x = np.array([[2., 10.], [2., 15.], [2., 5.], [4., 10.]])
        self.assertEqual(predict(frozen, x).tolist(), [False, True, False, False])
        # Inference of a question is independent of the other questions present.
        self.assertEqual(predict(frozen, x[1:2]).tolist(), [True])


if __name__ == '__main__':
    unittest.main()
