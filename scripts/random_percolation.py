# script to run random percolation on simulated graph database
# creates dataframes with data on the percolated graphs
from src.SimulatedReactionGraph import SimulatedReactionGraph
from src import constants
from scripts.make_graphs import load_solution_to_graph
import torinanet as tn
import os
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import signal

SOLVER_TIMEOUT = 5 * 60
MAX_PERCOLATION_TRAILS = 10

def get_graph_files(graph_dir: str):
    """Generator for all graphs in data dir. returns a SimulatedGraph object and rep no."""
    for parent in os.listdir(graph_dir):
        parent = os.path.join(graph_dir, parent)
        for jfile in os.listdir(parent):
            yield os.path.join(parent, jfile)

def percolate_reactions(g: tn.core.RxnGraph, n: int):
    # counter = 0
    # ajr = g
    # while counter < n:
    #     if ajr.get_n_reactions() == 0:
    #         break
    if n > g.get_n_reactions():
        return tn.core.RxnGraph()
    nodes = np.random.choice(list(g.reactions), n, replace=False)
    return g.remove_reactions(nodes)
    # ajr = ajr.remove_reactions(nodes)
    # counter += n
    # counter += 1
    # return ajr

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Kinetics solving timed out")

signal.signal(signal.SIGALRM, timeout_handler)

def solve_kinetics(g: tn.core.RxnGraph, **kwargs):
    signal.alarm(SOLVER_TIMEOUT) 
    try:
        solver = tn.analyze.kinetics.KineticAnalyzer(g, "k")
        iconcs = [1 if s in g.source_species else 0 for s in g.species]
        solver.solve_kinetics(iconcs, **kwargs)w
        signal.alarm(0)  # Cancel the alarm if successful
        return solver
    except TimeoutException:
        return None

def compare_concentrations(odf: pd.DataFrame, rdf: pd.DataFrame) -> dict:
    """Compare different kinetic solveres of original (o) and reduced/percolated (r) graphs"""
    # takes the concentration dfs of original and reduced nets
    # compare concentration values - profile + ss value
    profile_err = 0
    ss_err = 0
    for sp in rdf.columns:
        f = interp1d(rdf.index, rdf[sp], bounds_error=False)
        profile_err += np.mean(np.abs(odf[sp] - f(odf.index)))
        oss = odf[sp].iloc[-1]
        rss = rdf[sp].iloc[-1]
        ss_err += np.abs(oss - rss)
    # calculate average
    nsps = len(rdf.columns)
    profile_err = profile_err / nsps
    ss_err = ss_err / nsps
    # normalize by average conc values
    return {"conc_profile_error": profile_err, "conc_ss_error": ss_err}

def compare_rates(odf: pd.DataFrame, rdf: pd.DataFrame) -> dict:
    # takes the rates dataframe of the original net and reduced net
    # compare concentration values - profile + ss value
    profile_err = 0
    ss_err = 0
    for rxn in rdf.columns:
        f = interp1d(rdf.index, rdf[rxn], bounds_error=False)
        profile_err += np.mean(np.abs(odf[rxn] - f(odf.index)))
        oss = odf[rxn].max()
        rss = rdf[rxn].max()
        ss_err += np.abs(oss - rss)
    # calculate average
    nsps = len(rdf.columns)
    profile_err = profile_err / nsps
    ss_err = ss_err / nsps
    # normalize by average conc values
    return {"rate_profile_error": profile_err, "max_rate_error": ss_err}


def covered_rate(odf: pd.DataFrame, rdf: pd.DataFrame) -> dict:
    covered = odf.loc[:, rdf.columns].sum(axis=1).values / odf.sum(axis=1).values
    res = {"min_coverage": np.min(covered), "mean_coverage": np.mean(covered), "ss_coverage": covered[-1]}
    res["max_rate_coverage"] = odf.max(axis=0)[rdf.columns].sum() / odf.max(axis=0).sum()
    return res


def concs_df(g: tn.core.RxnGraph):
    data = {}
    for sp in g.species:
        data[g.specie_collection.get_key(sp)] = sp.properties["concentration"]
    return pd.DataFrame(data, index=g.properties["simulation_time"])

def rates_df(g: tn.core.RxnGraph):
    data = {}
    for rxn in g.reactions:
        data[g.reaction_collection.get_key(rxn)] = rxn.properties["rate"]
    return pd.DataFrame(data, index=g.properties["simulation_time"])


def analyze_graph(og: tn.core.RxnGraph, rg: tn.core.RxnGraph, solver_kwargs: dict):
    """export key metrics of graph to a dict"""
    conc_odf = concs_df(og)
    rates_odf = rates_df(og)
    solver = solve_kinetics(rg, **solver_kwargs)
    # check for timeout
    if solver is None:
        return None
    conc_rdf = solver.concentrations_df()
    rates_rdf = solver.rates_df()
    # check for negative concentrations
    if conc_rdf.values.min() < - solver_kwargs["atol"]:
        return None
    ajr = {
        "rg_reactions": rg.get_n_reactions(),
        "rg_species": rg.get_n_species(),
    }
    ajr.update(compare_concentrations(conc_odf, conc_rdf))
    ajr.update(compare_rates(rates_odf, rates_rdf))
    ajr.update(covered_rate(rates_odf, rates_rdf))
    return ajr


def single_run(path: str, n_percolation_steps: int, percolation_rep: int):
    og = SimulatedReactionGraph.from_file(path)
    solver_kwargs = {k: v for k, v in og.properties.items() if not "time" in k} # use the same tolerances as simulated graph
    og = og.to_rxngraph()
    rep = int(os.path.split(path)[-1].split(".")[0])
    percolation_step_size = int(round(og.get_n_reactions() / n_percolation_steps))
    details = {"og_reactions": og.get_n_reactions(), "og_species": og.get_n_species(), "rep": rep, "percolation_rep": percolation_rep, "percolation_step_size": percolation_step_size, "total_percolated": 0}
    total_percolated = 0
    ntrails = 0
    res = []
    rg = og
    while True:
        ajr = percolate_reactions(rg, percolation_step_size)
        # if the percolated graph is empty - stop run and return current results
        if ajr.get_n_reactions() == 0:
            return res
        d = analyze_graph(og, ajr, solver_kwargs)
        if d is not None:
            ntrails = 0
            rg = ajr
            d.update(details)
            total_percolated += percolation_step_size
            d["total_percolated"] = total_percolated
            res.append(d)
        else:
            ntrails += 1
        if ntrails > MAX_PERCOLATION_TRAILS:
            return res

def safe_single_run(args):
    path, n_percolation_steps, percolation_rep = args
    # this is essential for ensuring random behavoir in parallel processing
    np.random.seed(percolation_rep)
    # try:
    return single_run(path, n_percolation_steps, percolation_rep)
    # except Exception as e:
    #     print(f"Error with {path}, rep {percolation_rep}: {e}")
    #     return []

if __name__ == "__main__":
    args = []
    n_percolation_steps = 100
    n_percolation_reps = 2
    for path in get_graph_files(os.path.join(constants.DATA_DIR, "simulated")):
        for i in range(n_percolation_reps):
            args.append((path, n_percolation_steps, i + 1))
    data = []
    with Pool(processes=6) as pool:
        for result in tqdm(pool.imap_unordered(safe_single_run, args), total=len(args)):
            data.extend(result)
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(constants.RESULTS_DIR, "percolation.csv"))
