import unittest
import numpy as np
import esr
from esr import SourceSimulator, PointSourceSimulator, SpreadSourceSimulator
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
    model = esr.model.Model(cortex, forward_df)
    model.cortex = cortex
    model.forward = forward_df

    # Add a RAS transform, required for FreeSurfer.
    transform = AffineTransform(CoordinateSystem.RAS, np.eye(4))
    model.cortex.add_transform(transform)
    model.cortex.transform_to(CoordinateSystem.RAS)

    return model


class TestSourceSimulator(unittest.TestCase):
    """Test the esr.SourceSimulator class."""

    def test_init(self):
        """Test the __init__ method."""

        # Create a simple model.
        model = minimal_model()
        sources = [0,1,2,3]
        waveform = np.sin(np.linspace(0, 2*np.pi, 1000))

        SS = SourceSimulator(model, sources, waveform)

        self.assertEqual(SS.model, model)
        self.assertEqual(SS.sources, sources)
        np.testing.assert_array_almost_equal(SS.waveform, waveform)

        # Sources must be an integer.
        incorrect_sources = 1.5
        self.assertRaises(ValueError, SourceSimulator, model,
                          incorrect_sources, waveform)
        # If sources is a numpy array, it must be 1D.
        incorrect_sources = np.array(sources).reshape((2,2))
        self.assertRaises(ValueError, SourceSimulator, model,
                          incorrect_sources, waveform)
        # If sources is a numpy array, all elements must be integers.
        incorrect_sources = np.asarray(sources).astype(float)
        self.assertRaises(ValueError, SourceSimulator, model,
                          incorrect_sources, waveform)
        # If sources is a list, all elements must be integers.
        incorrect_sources = sources.copy()
        incorrect_sources[1] = incorrect_sources[1]*1.5
        self.assertRaises(ValueError, SourceSimulator, model,
                          incorrect_sources, waveform)

        # If waveform is a numpy array, it must be 1D.
        incorrect_waveform = np.array([waveform, waveform])
        self.assertRaises(ValueError, SourceSimulator, model,
                          sources, incorrect_waveform)

    def test_simulate(self):
        """Test if the simulate method is implemented."""

        self.assertRaises(NotImplementedError, SourceSimulator.simulate, self)


class TestPointSourceSimulator(unittest.TestCase):
    """Test the esr.PointSourceSimulator class."""

    def test_simulate(self):
        """Test the simulate method."""

        model = minimal_model()
        sources = [0, 1]
        waveform = np.ones(10)
        T = len(waveform)
        X = np.zeros((model.cortex.nb_vertices, T))
        X[0:2, :] = waveform

        PSS = PointSourceSimulator(model, sources=sources, waveform=waveform)
        X_pss = PSS.simulate()
        np.testing.assert_array_almost_equal(X_pss, X)


class TestSpreadSourceSimulator(unittest.TestCase):
    """Test the esr.SpreadSourceSimulator class."""

    def test_init(self):
        """Test the init method."""

        model = minimal_model()
        sources = [0]
        waveform = np.ones(10)
        decay = np.array([1, 0.7, 0.3])

        # If decay is a numpy array, it must be 1D.
        incorrect_decay = decay.reshape((3,1))
        self.assertRaises(ValueError, SpreadSourceSimulator, model, sources,
                          waveform, incorrect_decay)
        # Decay must be in descending order.
        incorrect_decay = decay[::-1]
        self.assertRaises(ValueError, SpreadSourceSimulator, model, sources,
                          waveform, incorrect_decay)
        incorrect_decay = decay[::-1].tolist()
        self.assertRaises(ValueError, SpreadSourceSimulator, model, sources,
                          waveform, incorrect_decay)

    def test_simulate(self):
        """Test the simulate method. """

        model = minimal_model()
        tri = model.cortex.triangles
        sources = [0]
        waveform = np.ones(10)
        decay = np.array([1, 0.7, 0.3])
        T = len(waveform)
        X = np.zeros((model.cortex.nb_vertices, T))
        X[0, :] = decay[0]
        X[1, :] = decay[1]
        X[2, :] = decay[1]
        X[3, :] = decay[2]

        SSS = SpreadSourceSimulator(model, sources=sources,waveform=waveform,
                                    decay=decay)
        X_sss = SSS.simulate()
        np.testing.assert_array_almost_equal(X_sss, X)

