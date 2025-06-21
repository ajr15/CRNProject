# make graphs to be used as database for testing different reduction algorithms
import os
from scipy.stats import norm
from multiprocessing import Pool
from itertools import product
import numpy as np
from model import PreferentialAttachmentModel, KineticSolver, SimulatedReactionGraph
from tqdm import tqdm

SPECIE_ENERGY_DIST = norm(0.4, 0.3)  # Default specie energy distribution
TEMPERATURE = 700  # Default temperature in Kelvin
MIN_PARTICIPATION = 0.9  # Minimum participation threshold for reactions

def solve_kinetics(g: SimulatedReactionGraph):
    solver = KineticSolver(g, "k")
    g.set_origin()
    iconcs = [1 if s in s == g.origin else 0 for s in g.get_species()]
    step = 1e-13
    solver.solve_kinetics(100 * step, step, iconcs)
    return solver

def generate_network(n_species, n_reactions, sp_energy_dist, temperature):
    model = PreferentialAttachmentModel(n_species=n_species, 
                                            n_reactions=n_reactions, 
                                            specie_energy_dist=sp_energy_dist)
    g = model.generate_graph()
    log_difference_distribution = norm(0, 1.105686e4 / temperature - 7.943808e-1) # from konnov data
    g.add_kinetic_data(temperature, log_difference_distribution)
    # solve kinetics of graph and add data to it
    solution = solve_kinetics(g)
    concs = solution.concentrations_df()
    rates = solution.get_rates()
    total_rates = rates.sum(axis=1)
    relative_rates = rates.apply(lambda x: x / total_rates, axis=0)
    for s in g.get_species():
        g.get_properties(s)["concentration"] = concs[s].tolist()
    for r in g.get_reactions():
        g.get_properties(r)["rate"] = rates[r].tolist()
        g.get_properties(r)["relative_rate"] = relative_rates[r].tolist()
    return g

def calculate_participation(g: SimulatedReactionGraph):
    participation = None
    for r in g.get_reactions():
        rate = np.array(g.get_properties(r)["rate"])
        p = np.where(rate > 0, 1, 0)
        if participation is None:
            participation = p
        else:
            participation += p
    return np.mean(participation) / len(g.get_reactions())

def single_run(args):
    nspecies, nreactions, rep = args
    g = generate_network(nspecies, nreactions, SPECIE_ENERGY_DIST, TEMPERATURE)
    participation = calculate_participation(g)
    if participation > MIN_PARTICIPATION:
        g.save("nets/s{}r{}/{}.json".format(nspecies, nreactions, rep))

if __name__ == "__main__":
    nspecies = [25, 50, 100]
    nreactions = [1000, 2500, 5000]
    # generage direcotories for saving networks
    for s, r in product(nspecies, nreactions):
        dir_name = "nets/s{}r{}".format(s, r)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    nreps = 50
    args = [(s, r, rep) for s, r, rep in product(nspecies, nreactions, range(nreps))]
    with Pool(processes=8) as pool:
        # use tqdm to show progress bar for multiprocessing
        for _ in tqdm(pool.imap_unordered(single_run, args), total=len(args)):
            pass