import numpy as np
from os import listdir
from os.path import join
import nimesh
from nimesh import Mesh, CoordinateSystem
import pandas as pd

class Model:
    
    def __init__(self, cortex, forward):
        """
        Args:
            cortex : Mesh object.
            forward : Forward operator (gain matrix).
        Raises:
            TypeError: Cortex must be a Mesh object.
                       Forward must be convertible to a numpy array of floats.
            ValueError: Number of columns in the forward operator has to be
            the same as the number of vertices in the mesh.
        """
        if not isinstance(cortex, Mesh):
            raise TypeError('\'cortex\' must be a Mesh object.')

        elif not isinstance(forward, np.ndarray):
            try:
                forward = np.array(forward, dtype = np.float64)
            except Exception:
                raise TypeError('\'forward\' must be convertible to a numpy'
                                'array of floats. ')

        # Raise an error if cortex and forward are not consistent.
        if len(np.asarray(cortex.vertices)) != forward.shape[1]:
            raise ValueError("Number of columns in forward operator must be "
                             "the same as the number of vertices in the mesh.")
        self.cortex = cortex
        self.forward = forward
    

def load_model(model_path):
        """
        Loads the files from model directory (this directory should contain
        cortex in GIfTI (.gii) format and forward operator in HDF format.
        Args:
             model_path: The path to the model's directory.
        Raises:
             TypeError: If cortex or forward operator don't exist or they 
             are not in the correct format (.gii for cortex and HDF for
             forward operator).
        """
        cortex = None
        forward = None

        for file in listdir(model_path):
            if file.endswith(".gii"):    
                cortex_path = join(model_path, file)
                cortex = nimesh.io.load(cortex_path)
                
            elif file.endswith((".h5",".h4",".hdf",".hdf4",".hdf5",".he2",
                                ".he5")):
                forward_path = join(model_path, file)
                forward = pd.read_hdf(forward_path).as_matrix()

        if cortex is None:
            raise TypeError("Cortex doesn't exist or it is not in the "
                            "GIfTI format (.gii).")

        if forward is None:
            raise TypeError("Forward operator doesn't exist or it is "
                            "not in the HDF format (.h4, .h5, .hdf., "
                            ".hdf4, .hdf5, .he2, .he5)")

        model = Model(cortex, forward)

        return model


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