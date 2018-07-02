import unittest
import numpy as np
from esr import Metric, L2norm


class TestMetrics(unittest.TestCase):
    """Test the esr.Metric class."""

    def test_init(self):
        """Test the init method."""

        src_amps = np.ones((5,5))
        est_src_amps = 0.7 * np.ones((5,5))

        # Create an instance of class Metric.
        metric = Metric(src_amps, est_src_amps)

        np.testing.assert_array_almost_equal(metric.src_amp, src_amps)
        np.testing.assert_array_almost_equal(metric.est_src_amp, est_src_amps)

        # src_amplitudes must be convertible to a numpy array of floats.
        incorrect_src_amps = '1 2 3'
        self.assertRaises(TypeError, Metric, incorrect_src_amps, est_src_amps)
        # Shape of src_amp and est_src_amp must be equal.
        incorrect_src_amps = 5
        self.assertRaises(ValueError, Metric, incorrect_src_amps, est_src_amps)

    def test_calculate(self):
        """Test if the calculate method is implemented."""

        self.assertRaises(NotImplementedError, Metric.calculate, self)


class TestL2norm(unittest.TestCase):

    def test_calculate(self):
        """Test the calculate method."""

        src_amps = np.ones((5,5))
        est_src_amps = 0.7 * np.ones((5,5))
        err = 0
        for i in range(0, src_amps.shape[0]):
            for j in range(0, src_amps.shape[1]):
                err += (src_amps[i,j] - est_src_amps[i,j])**2

        err = np.sqrt(err)
        self.assertEqual(L2norm(src_amps, est_src_amps).calculate(), err)
