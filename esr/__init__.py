from esr.model import Model
from esr.model import load_model
from os import listdir
from os.path import join
import nimesh
import pandas as pd
import numpy as np
from esr.inverse_methods import InverseMethod
from esr.inverse_methods import PseudoInverse
from esr.inverse_methods import TikhonovInverse


def load_models(models_path):
    """
    Loads the files from model directory (this directory should contain
    cortex in GIfTI (.gii) format and forward operator in HDF format.
    Args:
         model_path: The path to the model's directory.
    Raises:
         TypeError: If cortex or forward operator don't exist or they
         are not in the correct format (.gii for cortex and HDF for
         forward operator)
         ValueError: Number of columns in forward operator has to be
         the same as the number of vertices in the mesh
    """

    all_models = []
    models_path = "/user/ikojcic/home/src/MARS/models_dir"
    all_paths = [join(models_path, model_id) for model_id in listdir(models_path)]
    # model_id ='model1', 'model2',...


    for model_path in all_paths:
        model = Model()
        for file in listdir(model_path):
            if file.endswith(".gii"):
                cortex_path = join(model_path, file)
                model.cortex = nimesh.io.load(cortex_path)

            elif file.endswith((".h5", ".h4", ".hdf", ".hdf4", ".hdf5", ".he2",
                                ".he5")):
                forward_path = join(model_path, file)
                model.forward = pd.read_hdf(forward_path).as_matrix()


        if model.cortex is None:
            raise TypeError("Cortex doesn't exist or it is not in the "
                            "GIfTI format (.gii).")

        if model.forward is None:
            raise TypeError("Forward operator doesn't exist or it is "
                            "not in the HDF format (.h4, .h5, .hdf., "
                            ".hdf4, .hdf5, .he2, .he5)")

        # Raise an error if cortex and forward are not consistent
        if len(np.asarray(model.cortex.vertices)) != model.forward.shape[1]:
            raise ValueError("Number of columns in forward operator must be "
                             "the same as the number of vertices in the mesh.")

        all_models.append(model)

    return all_models