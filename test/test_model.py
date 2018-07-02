import unittest
import numpy as np
import esr
import tempfile
import os
import nimesh
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


class TestModel(unittest.TestCase):
    """ Test the Model class from esr.model module."""

    def test_init(self):
        """Test the __init__ method."""

        cortex = minimal_mesh()
        forward = np.eye(4)

        # The number of columns in forward operator must be the same
        # as the number of vertices in the mesh.
        self.assertEqual(len(np.asarray(cortex.vertices)), forward.shape[1])

        # Cortex must be a Mesh object.
        incorrect_cortex = np.array([1,2,3])
        self.assertRaises(TypeError, esr.model.Model, incorrect_cortex, forward)

        # Forward must be convertible to a numpy array.
        incorrect_forward = ['hello']
        self.assertRaises(TypeError, esr.model.Model, cortex, incorrect_forward)


class TestLoadingModel(unittest.TestCase):

    def test_load_model(self):
        """Test the function load_model from esr.model module."""

        cortex = minimal_mesh()
        forward = np.array(np.ones((3,4)))
        forward_df = pd.DataFrame(forward)

        model = esr.model.Model(cortex, forward_df)

        transform = AffineTransform(CoordinateSystem.RAS, np.eye(4))
        cortex.add_transform(transform)
        cortex.transform_to(CoordinateSystem.RAS)

        with tempfile.TemporaryDirectory() as directory:

            cortex_name = os.path.join(directory, 'c.gii')
            forward_name = os.path.join(directory, 'h.h5')

            # Save and reload the test data.
            nimesh.io.save(cortex_name, model.cortex)
            model.forward = pd.DataFrame(forward)
            model.forward.to_hdf(forward_name, key='df', mode='w')

            loaded_model = esr.load_model(directory)

            self.assertEqual(model.cortex.nb_vertices,
                             loaded_model.cortex.nb_vertices)
            self.assertEqual(model.cortex.nb_triangles,
                             loaded_model.cortex.nb_triangles)

            np.testing.assert_array_almost_equal(model.forward,
                                                 loaded_model.forward)

        with tempfile.TemporaryDirectory() as directory:
            # No .gii or hdf files.
            # If load_model function fails to be executed then this test
            # is successful.
            self.assertRaises(TypeError, esr.load_model, directory)


class TestLoadingModels(unittest.TestCase):

    def test_load_models(self):
        """Test the function load_models from the __init__ method."""

        cortex = minimal_mesh()
        forward = np.array(np.ones((3, 4)))
        forward_df = pd.DataFrame(forward)

        model = esr.model.Model(cortex, forward_df)

        transform = AffineTransform(CoordinateSystem.RAS, np.eye(4))
        cortex.add_transform(transform)
        cortex.transform_to(CoordinateSystem.RAS)

        all_models = []

        with tempfile.TemporaryDirectory() as directory:

            for i in range(2):
                suf = 'example' + str(i)
                subdir = tempfile.mkdtemp(suffix=suf, dir=directory)
                os.chdir(subdir)

                cortex_name = os.path.join(subdir, 'c.gii')
                forward_name = os.path.join(subdir, 'h.h5')

                # Save and reload the test data.
                nimesh.io.save(cortex_name, model.cortex)
                model.forward = pd.DataFrame(forward)
                model.forward.to_hdf(forward_name, key='df', mode='w')

                all_models.append(model)


            loaded_models = esr.load_models(directory)


        for i in range(2):
            self.assertEqual(all_models[i].cortex.nb_vertices,
                             loaded_models[i].cortex.nb_vertices)
            self.assertEqual(all_models[i].cortex.nb_triangles,
                             loaded_models[i].cortex.nb_triangles)

            np.testing.assert_array_almost_equal(all_models[i].forward,
                                                 loaded_models[i].forward)
