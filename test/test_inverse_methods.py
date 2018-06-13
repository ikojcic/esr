import unittest
import numpy as np
import esr
from esr import InverseMethod, PseudoInverse, TikhonovInverse
import pandas as pd
from nimesh import AffineTransform, CoordinateSystem, Mesh
from numpy import linalg as LA

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
    forward = np.eye(4)
    forward_df = pd.DataFrame(forward)
    model = esr.model.Model()
    model.cortex = cortex
    model.forward = forward_df

    # Add a RAS transform, required for FreeSurfer.
    transform = AffineTransform(CoordinateSystem.RAS, np.eye(4))
    model.cortex.add_transform(transform)
    model.cortex.transform_to(CoordinateSystem.RAS)

    return model


class FakeModel:

    def __init__(self, cortex=None, forward=None, a=None):
        self.cortex = cortex
        self.forward = forward
        self.a = a


def simple_noisy_measurements():
    """Returns simulated noisy measurements for point source simulation."""

    model = minimal_model()
    G = model.forward
    t = np.linspace(0, 2 * np.pi, 1000)
    waveform = 10 * np.sin(t)
    sources = [0, 1, 2]
    X_pss = np.zeros((model.cortex.nb_vertices, len(t)))
    X_pss[sources, :] = waveform
    m = np.dot(G, X_pss)

    SNR = 15
    varS = np.var(m)
    varN = varS / SNR
    sigmaN = np.sqrt(varN)
    nb_sensors = m.shape[0]
    T = m.shape[1]
    N = np.random.normal(0, sigmaN, [nb_sensors, T])
    M = m + N

    return M


class TestInverseMethod(unittest.TestCase):
    """Test esr.InverseMethod class."""

    def test_init(self):
        """Test the __init__ method."""

        # Create simple model and measurements.
        model = minimal_model()
        M = simple_noisy_measurements()
        IM = InverseMethod(model, M)

        self.assertEqual(IM.model, model)
        np.testing.assert_array_almost_equal(IM.M, M)

        # Model must be an instance of class Model.
        incorrect_model = FakeModel()
        self.assertRaises(TypeError, InverseMethod, incorrect_model, M)

        # Number of rows in the forward operator has to match the number of
        # rows in M.
        self.assertEqual(model.forward.shape[0], M.shape[0])

        incorrect_M = M.copy().reshape((8, 500))
        self.assertRaises(ValueError, InverseMethod, model, incorrect_M)

    def test_compute(self):
        """Test if the compute method is implemented."""

        self.assertRaises(NotImplementedError, InverseMethod.compute, self)


class TestPseudoInverse(unittest.TestCase):
    """Test esr.PseudoInverse class."""

    def test_compute(self):
        """Test the compute method."""

        # Create simple forward operator and measurements.
        model = minimal_model()
        M = simple_noisy_measurements()
        G = np.asarray(model.forward)
        Gp = np.dot(G.T, LA.inv(np.dot(G, G.T)))
        Xh = np.dot(Gp, M)

        PI = PseudoInverse(model, M)
        Xhat = PI.compute()
        np.testing.assert_array_almost_equal(Xh, Xhat)

        # Check for a well-conditioned system.
        X = np.random.rand(4,10)
        M2 = np.dot(G,X)
        PI2 = PseudoInverse(model, M2)
        Xhat2 =  PI2.compute()
        np.testing.assert_array_almost_equal(X, Xhat2)



class TestTikhonovInverse(unittest.TestCase):
    """Test esr.TikhonovInverse class."""

    def test_compute(self):
        """Test the compute method."""

        model = minimal_model()
        M = simple_noisy_measurements()
        G = np.asarray(model.forward)

        # When G is close to singular and then regularized, estimated source
        # amplitudes should be equal to the ones obtained with
        # TikhonovInverse.compute().
        G1 = G.copy()
        G1[-1,-1] = 10**-6
        A = np.dot(G1, G1.T)
        optimal_lam = 10**-3
        G1t = np.dot(G1.T, LA.pinv(A + (optimal_lam ** 2) * np.eye(G.shape[0])))
        Xh1 = np.dot(G1t, M)

        TI = TikhonovInverse(model, M)
        Xht = TI.compute()
        np.testing.assert_allclose(Xh1, Xht, atol=1e+1)

        # When lambda = 0, Tikhonov solution should be equal to pseudoinverse
        # solution. It is expected that optimal lambda = 0 when G is an identity
        #  matrix (no need for regularization).
        PI = PseudoInverse(model, M)
        Xhp = PI.compute()
        np.testing.assert_allclose(Xht, Xhp, atol=1e+1)

        # When lambda is very high (approaches infinity), when G is an identity
        # matrix, it is expected that the matrix (GG' + lambda^2*I)^-1 has all
        # elements close to 0.
        A = np.dot(G, G.T)
        optimal_lam = 10**+5
        G2t = np.dot(G1.T, LA.pinv(A + (optimal_lam ** 2) * np.eye(G.shape[0])))
        Xh2 = np.dot(G2t, M)

        X_expected = np.zeros(Xh2.shape)
        np.testing.assert_allclose(Xh2, X_expected, atol=1e-5)