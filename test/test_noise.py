import unittest
import numpy as np
import esr
from esr import NoiseSimulator, GaussianNoiseSimulator, NoiselessSimulator
import pandas as pd
from nimesh import AffineTransform, CoordinateSystem, Mesh


def minimal_mesh():
    """Returns a mesh with two triangles."""

    # Create a simple mesh with two triangles.
    vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
    ]

    triangles = [
        [0, 1, 2],
        [1, 2, 3],
    ]

    return Mesh(vertices, triangles, CoordinateSystem.SCANNER)


def minimal_model():
    """Returns an instance of class Model."""

    cortex = minimal_mesh()
    forward = np.array(np.ones((3, 4)))
    forward_df = pd.DataFrame(forward)
    model = esr.model.Model()
    model.cortex = cortex
    model.forward = forward_df

    # Add a RAS transform, required for FreeSurfer.
    transform = AffineTransform(CoordinateSystem.RAS, np.eye(4))
    model.cortex.add_transform(transform)
    model.cortex.transform_to(CoordinateSystem.RAS)

    return model

def simple_measurements():
    """Returns simulated noiseless measurements for point source simulation."""

    model = minimal_model()
    G = model.forward
    t = np.linspace(0, 2 * np.pi, 1000)
    waveform = 10 * np.sin(t)
    sources = [0, 1, 2]
    X_pss = np.zeros((model.cortex.nb_vertices, len(t)))
    X_pss[sources, :] = waveform
    m = np.dot(G, X_pss)

    return m


class TestNoiseSimulator(unittest.TestCase):
    """Test the esr.NoiseSimulator class"""

    def test_init(self):
        """Test the __init__ method."""

        # Create simple measurements
        m = simple_measurements()
        noise_model = NoiseSimulator(m)
        np.testing.assert_array_almost_equal(noise_model.m, m)

        # m must be a numpy array of floats.
        self.assertRaises(TypeError, NoiseSimulator, None)

    def test_add_noise(self):
        """Test if the add_noise method is implemented."""
        self.assertRaises(NotImplementedError, NoiseSimulator.add_noise,
                          self)


class TestGaussianNoiseSimulator(unittest.TestCase):

    def test_init(self):
        """Test the init method"""

        m = simple_measurements()
        SNR = 15
        gaussian_noise_model = GaussianNoiseSimulator(m, SNR = SNR)
        self.assertEqual(gaussian_noise_model.SNR, SNR)

        # SNR must be an integer or a float.
        incorrect_SNR = 'ten'
        self.assertRaises(ValueError, GaussianNoiseSimulator, m, incorrect_SNR)

    def test_add_noise(self):
        """Test the add_noise method. """

        m = simple_measurements()
        SNR = 15
        gaussian_noise_model = GaussianNoiseSimulator(m, SNR = SNR)
        noisy_m = gaussian_noise_model.add_noise()
        noise = noisy_m - m

        # Test the properties of the noise.
        varN = np.var(noise)
        expected_varN = np.var(m)/SNR
        np.testing.assert_allclose(varN, expected_varN,  atol=1e+1)


class TestNoiselessSimulator(unittest.TestCase):
    """Test the add_noise method."""

    def test_add_noise(self):
        m = simple_measurements()
        noiseless_model = NoiselessSimulator(m)
        np.testing.assert_array_almost_equal(noiseless_model.m, m)
