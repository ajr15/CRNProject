# make graphs to be used as database for testing different reduction algorithms
import os
from scipy.stats import norm
from multiprocessing import Pool
from itertools import product
import numpy as np
import torinanet as tn
from src.GeneratingModels import PreferentialAttachmentModel, SimulatedReactionGraph
from src.KineticSolver import KineticSolver
from tqdm import tqdm
import signal

SPECIE_ENERGY_DIST = norm(-1620.6, 265229.3)  # Default specie energy distribution
TEMPERATURE = 1000  # Default temperature in Kelvin
GAS_CONSTANT = 8.314  # J / K mol
SOLVER_ATOL = 1e-25
SOLVER_SS_LIMIT = 1e-6
SOLVER_RTOL = 1e-3
SOLVER_TIMEOUT = 60 * 5 # timeout after 5 minutes

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Kinetics solving timed out")

signal.signal(signal.SIGALRM, timeout_handler)

def solve_kinetics(g: SimulatedReactionGraph, ss_threshold=SOLVER_SS_LIMIT, atol=SOLVER_ATOL, rtol=SOLVER_RTOL):
    signal.alarm(SOLVER_TIMEOUT) 
    try:
        solver = KineticSolver(g, "k")
        iconcs = [1 if s in g.origin else 0 for s in g.get_species()]
        solver.solve_kinetics(iconcs, ss_threshold=ss_threshold, atol=atol, rtol=rtol)
        signal.alarm(0)  # Cancel the alarm if successful
        return solver
    except TimeoutException:
        return None

def add_rate_constants(g: SimulatedReactionGraph, temp: float):
    for r in g.reactions():
        props = r.properties
        # Calculate the rate constant using the Arrhenius equation
        r.properties["k"] = props["A"] * np.power(temp, props["beta"]) * np.exp(- props["Ea"] / (GAS_CONSTANT * temp))
    return g

def load_solution_to_graph(g: SimulatedReactionGraph, solver: KineticSolver) -> SimulatedReactionGraph:
    g.properties["simulation_time"] = solver._ts.tolist()
    g.properties["ss_threshold"]=SOLVER_SS_LIMIT
    g.properties["atol"]=SOLVER_ATOL
    g.properties["rtol"]=SOLVER_RTOL
    concs = solver.concentrations_df()
    rates = solver.rates_df()
    total_rates = rates.sum(axis=1)
    relative_rates = rates.apply(lambda x: x / total_rates, axis=0)
    for s in g.get_species():
        g.get_properties(s)["concentration"] = concs[s].values.tolist()
    for r in g.get_reactions():
        g.get_properties(r)["rate"] = rates[r].values.tolist()
        g.get_properties(r)["relative_rate"] = relative_rates[r].values.tolist()
    return g

def generate_network(n_species, n_reactions, sp_energy_dist, temperature):
    model = PreferentialAttachmentModel(n_species=n_species, 
                                        n_reactions=n_reactions, 
                                        specie_energy_dist=sp_energy_dist)
    g = model.generate_graph()
    g.origin = list(np.random.choice(g.get_species(), 2)) # set randomly two origin points
    g.add_kinetic_data()
    g = add_rate_constants(g, temperature)
    return g
    

def fully_connected(g: SimulatedReactionGraph) -> bool:
    """Make sure the graph is fully connected"""
    rxn_graph = g.to_rxngraph()
    sp = tn.analyze.algorithms.ShortestPathAnalyzer(rxn_graph, prop_func=lambda rxn: 1)
    return all(sp.shortest_path_table["dist"] < np.inf)

def single_run(args):
    nspecies, nreactions, rep, parent_dir = args
    # this is essential to make sure randomness occures also in parallel runs
    np.random.seed(rep)
    # try up to 10 times to generate a fully connected network
    for _ in range(10):
        g = generate_network(nspecies, nreactions, SPECIE_ENERGY_DIST, TEMPERATURE)
        if fully_connected(g):
            # solve kinetics of graph and add data to it
            solution = solve_kinetics(g)
            # if solver is None - we reached timeout, skip
            if solution is None:
                continue
            # if any concentration is negative, we skip this graph (tolerances are not enough)
            if solution.concentrations_df().values.min() < - SOLVER_ATOL:
                continue
            # if the max reaction rate is less the SS definition, skip the graph (bad luch with origin selection)
            if solution.rates_df().values.max() < SOLVER_SS_LIMIT:
                continue
            g = load_solution_to_graph(g, solution)
            g.save(parent_dir + "/s{}r{}/{}.json".format(nspecies, nreactions, rep))
            break

if __name__ == "__main__":
    nspecies = [25, 50, 100]
    nreactions = [1000, 2500, 5000]
    parent_dir = "data/simulated"
    # generage direcotories for saving networks
    for s, r in product(nspecies, nreactions):
        dir_name = parent_dir + "/s{}r{}".format(s, r)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    nreps = 5
    args = [(s, r, rep, parent_dir) for s, r, rep in product(nspecies, nreactions, range(nreps))]
    with Pool(processes=6) as pool:
        # use tqdm to show progress bar for multiprocessing
        for _ in tqdm(pool.imap_unordered(single_run, args), total=len(args)):
            pass