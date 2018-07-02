from esr.model import Model
from esr.model import load_model
from esr.model import minimal_mesh
from os import listdir
from os.path import join
from esr.source_simulation import SourceSimulator
from esr.source_simulation import PointSourceSimulator
from esr.source_simulation import SpreadSourceSimulator
from esr.noise_simulation import NoiseSimulator
from esr.noise_simulation import GaussianNoiseSimulator
from esr.noise_simulation import NoiselessSimulator


def load_models(models_path):
    """
    Loads the files from model directory (this directory should contain
    cortex in GIfTI (.gii) format and forward operator in HDF format.
    Args:
         model_path: The path to the model's directory.
    """

    all_models = []
    models_path = models_path
    all_paths = [join(models_path, model_id) for model_id in listdir(models_path)]

    for model_path in all_paths:
        model = load_model(model_path)
        all_models.append(model)

    return all_models
