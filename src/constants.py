from scipy.stats import norm
from src.Distribution import LogNormal

# PHYSICAL CONSTANTS

R = 8.314462618 # universal gas constant
BOLTZMANN = 1.3806e-23
PLANK = 6.62607e-34
CAL_TO_J = 4.184 # conversion between calories to jouls
Ha_TO_JMOL = 2625.5e3 # conversion between hartree to jouls/mol

# DEFAULT RUN PARAMETERS

DEFAULT_TEMPERATURE = 1000
KINETIC_SOLVER_KWARGS = {"atol": 1e-25, "ss_threshold": 1e-6, "rtol": 1e-3}

# BASIC PROPERTY DISTRIBUTIONS
# fitted for the default temperature above from real network data

SPECIE_ENERGY_DIST = norm(121404.8304, 197977.8793) 
PREEXPONTIAL_DIST = LogNormal(28.7386, 11.6402)
BETA_DIST = norm(0.3203, 1.2347)
EA_DIST_ARGUMENTS = {
    "E<0": LogNormal(10.4638, 1.3725),
    "a": 1.2551e+00,
    "b": 3.3500e+04,
    "error": norm(0, 2.6644e+05)
}

# PATH CONSTANTS

import os

PARENT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(PARENT_DIR, "data")
RESULTS_DIR = os.path.join(PARENT_DIR, "results")

# reading graphs constants

from data.literature.parsers.utils import DummyAcMatrix, DummyHashedCollection

GRAPH_NAMES = ["ammonia", "hydrogen", "methane"]
BASE_LITERATURE_GRAPH_PATH = os.path.join(DATA_DIR, "literature/$.rxn")
RXN_GRAPH_PARAMS = {
    "ac_matrix_type": DummyAcMatrix,
    "reaction_collection": DummyHashedCollection, 
    "specie_collection": DummyHashedCollection,
    "specie_collection_kwargs": {"kind": "species"},
    "reaction_collection_kwargs": {"kind": "reactions"}
}