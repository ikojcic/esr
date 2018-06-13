import numpy as np


class NoiseSimulator():

    def __init__(self, m):
        """
        Parent class for noise simulators.
        Args:
             m: Measurements (M/EEG signals).
        Raises:
             TypeError: m must be convertible to a numpy array of floats.
             NotImplementedError: If add_noise method is not implemented in
             the child class.
        """

        if m is None:
            raise TypeError('\'m\' cannot be None')

        try:
            m = np.array(m, dtype=np.float64)
        except Exception:
            raise TypeError('\'m\' must be convertible to a numpy array '
                            'of floats.')

        self.m = m

    def add_noise(self):
        raise NotImplementedError


class GaussianNoiseSimulator(NoiseSimulator):

    def __init__(self, m, SNR = None):
        """
        Gaussian noise simulator.
        Args:
             SNR: Signal-to-noise ratio. Must be an integer or a float.
        Raises:
             ValueError: If SNR is not an integer or a float.
        """
        if SNR is None:
            SNR = 10

        try:
            SNR = float(SNR)
        except Exception:
            raise ValueError('\'SNR\' must be convertible to a float.')

        super().__init__(m)
        self.SNR = SNR

    def add_noise(self):
        """Returns simulated measurements with additive white Gaussian noise."""

        varS = np.var(self.m)
        varN = varS/ self.SNR
        sigmaN = np.sqrt(varN)
        nb_sensors = self.m.shape[0]
        T = self.m.shape[1]
        N = np.random.normal(0, sigmaN, [nb_sensors, T])
        self.m = self.m + N
        return self.m


class NoiselessSimulator(NoiseSimulator):

    def add_noise(self):
        """Returns simulated noise-free measurements."""

        return self.m


