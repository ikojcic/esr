import numpy as np


class NoiseSimulator():

    def __init__(self, measurements):
        """
        Parent class for noise simulators.
        Args:
             measurements: Measurements (M/EEG signals).
        Raises:
             TypeError: measurements must be convertible to a numpy array of
             floats.
             NotImplementedError: If add_noise method is not implemented in
             the child class.
        """

        if not isinstance(measurements, np.ndarray):
            try:
                measurements = np.array(measurements, dtype=np.float64)
            except Exception:
                raise TypeError('\'measurements\' must be convertible to a numpy '
                                'array of floats.')

        self.measurements = measurements

    def add_noise(self):
        raise NotImplementedError


class GaussianNoiseSimulator(NoiseSimulator):

    def __init__(self, measurements, snr=10.0):
        """
        Gaussian noise simulator.
        Args:
             snr: Signal-to-noise ratio. Must be an integer or a float.
        Raises:
             ValueError: If SNR is not an integer or a float.
        """

        if isinstance(snr, list) and len(snr)==1:
            try:
                snr = float(snr[0])
            except Exception:
                raise ValueError('\'snr\' must be convertible to a float.')

        if not isinstance(snr, float):
            try:
                snr = float(snr)
            except Exception:
                raise ValueError('\'snr\' must be convertible to a float.')

        super().__init__(measurements)
        self.snr = snr

    def add_noise(self):
        """Returns simulated measurements with additive white Gaussian noise."""

        varS = np.var(self.measurements)
        varN = varS/ self.snr
        sigmaN = np.sqrt(varN)
        nb_sensors = self.measurements.shape[0]
        T = self.measurements.shape[1]
        N = np.random.normal(0, sigmaN, [nb_sensors, T])
        self.measurements = self.measurements + N
        return self.measurements


class NoiselessSimulator(NoiseSimulator):

    def add_noise(self):
        """Returns simulated noise-free measurements."""

        return self.measurements
