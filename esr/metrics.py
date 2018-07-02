import numpy as np
from numpy import linalg as LA


class Metric:

    def __init__(self, src_amp, est_src_amp):
        """
        Parent class for metrics used to evaluate the accuracy of source
        amplitude estimation.
        Args:
             src_amp: Simulated source amplitudes. Must be a sequence
             convertible to a numpy array of floats.
             est_src_amp: Estimated source amplitudes, obtained by one of the
             inverse methods. Must be a sequence convertible to a numpy array
             of floats.
        Raises:
             TypeError: If src_amp or est_src_amp are not numpy arrays, they
             must be sequences convertible to a numpy array of floats.
             ValueError: Shape of src_amp and est_src_amp must be equal.
             NotImplementedError: If calculate method is not implemented in
             the child class.

        """


        if isinstance(src_amp, np.ndarray) == False:
            try:
                src_amp = np.array(src_amp, dtype = np.float64)
            except Exception:
                raise TypeError('\'src_amp\' must be convertible to a numpy '
                                'array of floats.')

        if isinstance(est_src_amp, np.ndarray) == False:
            try:
                est_src_amp = np.array(est_src_amp, dtype=np.float64)
            except Exception:
                raise TypeError('\'est_src_amp\' must be convertible to a '
                                'numpy array of floats.')

        if src_amp.shape != est_src_amp.shape:
            raise ValueError("Shape of \'src_amp\' and \'est_src_amp\' must"
                             " be equal.")

        self.src_amp = src_amp
        self.est_src_amp = est_src_amp

    def calculate(self):
        raise NotImplementedError


class L2norm(Metric):
    """Eucledian distance (l2-norm) used for evaluation of accuracy of
    source localization."""

    def calculate(self):
        """Returns an error (l2-norm) between the actual source amplitudes and
        the estimated ones."""
        err = LA.norm(self.src_amp - self.est_src_amp)
        return err