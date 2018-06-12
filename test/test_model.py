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



class TestModels(unittest.TestCase):
    """Test the esr.model module"""

    def test_load_model(self):
        """Test the function load_model."""

        # Generate artificial mesh (cortex) & forward operator
        cortex = minimal_mesh()
        forward = np.array(np.ones((3,4)))
        forward_df = pd.DataFrame(forward)

        # Generate artificial model
        model = esr.model.Model()
        model.cortex = cortex
        model.forward = forward_df

        # Add a RAS transform, required for FreeSurfer.
        transform = AffineTransform(CoordinateSystem.RAS, np.eye(4))
        cortex.add_transform(transform)
        cortex.transform_to(CoordinateSystem.RAS)

        # Work in a temporary directory. This guarantees cleanup even on error.
        # with tempfile. TemporaryDirectory() as directory:
        with tempfile.TemporaryDirectory() as directory:

            cortex_name = os.path.join(directory, 'c.gii')
            forward_name = os.path.join(directory, 'h.h5')

            # Save and reload the test data.
            nimesh.io.save(cortex_name, model.cortex)
            model.forward.to_hdf(forward_name, key='df', mode='w')

            loaded_cortex = nimesh.io.load(cortex_name)
            loaded_forward = pd.read_hdf(forward_name).as_matrix()
            loaded_model = esr.load_model(directory)


            self.assertEqual(model.cortex.nb_vertices,
                             loaded_cortex.nb_vertices)
            self.assertEqual(model.cortex.nb_triangles,
                             loaded_cortex.nb_triangles)
            # The number of columns in forward operator must be the same
            # as the number of vertices in the mesh.
            self.assertEqual(len(np.asarray(loaded_cortex.vertices)),
                             loaded_forward.shape[1])
            np.testing.assert_array_almost_equal(model.forward, loaded_forward)

        with tempfile.TemporaryDirectory() as directory:
            # No .gii or hdf files.
            # If load_model function fails to be executed then this test
            # is successful.
            self.assertRaises(TypeError, esr.load_model, directory)
