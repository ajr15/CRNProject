# make graphs and save them to a database
import os
from multiprocessing import Pool
from itertools import product
import numpy as np
import networkx as nx
import torinanet as tn
from src import utils, constants
from src.SqlModels import species_to_sql, reactions_to_sql, graph_to_sql, make_database

def add_shortest_path_properties(g: tn.core.RxnGraph) -> tn.core.RxnGraph:
    dist_analyzer = tn.analyze.algorithms.ShortestPathAnalyzer(g, prop_func=lambda rxn: 1)
    energy_analyzer = tn.analyze.algorithms.ShortestPathAnalyzer(g, prop_func=lambda rxn: max(rxn.properties["energy"], 0))
    for sp in g.species:
        sid = g.specie_collection.get_key(sp)
        sp.properties["shortest_path_distance"] = dist_analyzer.shortest_path_table.loc[sid, "dist"]
        sp.properties["shortest_path_absolute_energy"] = energy_analyzer.shortest_path_table.loc[sid, "dist"]
        sp.properties["shortest_path_total_energy"] = sum([r.properties["energy"] for r in energy_analyzer.get_path_to_source(sp)])
    return g

def load_molrank_properties(rxn_graph: tn.core.RxnGraph, g: nx.DiGraph, prefix: str) -> tn.core.RxnGraph:
    for rxn in rxn_graph.reactions:
        rxn.properties[prefix + "rate"] = g.nodes[rxn_graph.reaction_collection.get_key(rxn)]["rate"]
    for sp in rxn_graph.species:
        sp.properties[prefix + "conc"] = g.nodes[rxn_graph.specie_collection.get_key(sp)]["conc"]
    return rxn_graph

def add_molrank_properties(rxn_graph: tn.core.RxnGraph):
    # first run an "accurate" molrank, using the rate constant
    ranked = utils.molrank_ranking(rxn_graph, rate_constant_property="k")
    # load the data to the graph
    rxn_graph = load_molrank_properties(rxn_graph, ranked, "molrank_real_")
    # then run a thermodynamic estimate of molrank, using the EY rate constant
    ranked = utils.molrank_ranking(rxn_graph, rate_constant_property="k_ey")
    # load the data to the graph
    rxn_graph = load_molrank_properties(rxn_graph, ranked, "molrank_ey_")
    return rxn_graph

def add_kinetic_properties(rxn_graph: tn.core.RxnGraph, solver: tn.analyze.kinetics.KineticAnalyzer, prefix: str):
    # add reaction data
    rates = solver.rates_df()
    for rxn in rxn_graph.reactions:
        rid = rxn_graph.reaction_collection.get_key(rxn)
        rxn.properties[prefix + "max_rate"] = rates[rid].max()
        rxn.properties[prefix + "ss_rate"] = rates[rid].values[-1]
    # add specie data
    concs = solver.concentrations_df()
    for sp in rxn_graph.species:
        rid = rxn_graph.specie_collection.get_key(sp)
        sp.properties[prefix + "max_conc"] = concs[rid].max()
        sp.properties[prefix + "ss_conc"] = concs[rid].values[-1]
    return rxn_graph

def save_to_sql(g: tn.core.RxnGraph, gid: str, output_dir: str):
    # collect entries
    gentry = graph_to_sql(g, gid)
    gentry.kinetic_solver_args = constants.KINETIC_SOLVER_KWARGS
    entries = [gentry]
    entries += species_to_sql(g, gid)
    entries += reactions_to_sql(g, gid)
    # open a new sqlite database and commit the entries
    db_path = os.path.join(output_dir, f"{gid}.db")
    db = make_database(db_path)
    db.add_all(entries)
    db.commit()
    db.close()

def fully_connected(g: tn.core.RxnGraph) -> bool:
    """Make sure the graph is fully connected"""
    sp = tn.analyze.algorithms.ShortestPathAnalyzer(g, prop_func=lambda rxn: 1)
    return all(sp.shortest_path_table["dist"] < np.inf)


def single_run(nspecies: int, nreactions: int, rep: int, timeout: float, output_dir: str, ntrails: int):
    # this is essential to make sure randomness occures also in parallel runs
    np.random.seed(rep)
    # try up to 10 times to generate a fully connected network
    for _ in range(ntrails):
        print(_)
        g = utils.generate_network(nspecies, nreactions, constants.SPECIE_ENERGY_DIST)
        gid = "s{}r{}_{}".format(nspecies, nreactions, rep)
        if fully_connected(g):
            # solve kinetics of graph and add data to it - first solve accurately 
            solution = utils.solve_kinetics(g, timeout, constants.DEFAULT_TEMPERATURE, "k", **constants.KINETIC_SOLVER_KWARGS)
            # if solver is None - we reached timeout, skip
            if solution is None:
                continue
            # if any concentration is negative, we skip this graph (tolerances are not enough)
            if solution.concentrations_df().values.min() < - constants.KINETIC_SOLVER_KWARGS["atol"]:
                continue
            # if the max reaction rate is less the SS definition, skip the graph (bad luch with origin selection)
            if solution.rates_df().values.max() < constants.KINETIC_SOLVER_KWARGS["ss_threshold"]:
                continue
            # add kinetic properties (of ey as well)
            g = add_kinetic_properties(g, solution, "real_")
            ey_solution = utils.solve_kinetics(g, timeout, constants.DEFAULT_TEMPERATURE, "k_ey", **constants.KINETIC_SOLVER_KWARGS)
            # if computation is successful, add its properties
            if ey_solution is not None:
                g = add_kinetic_properties(g, ey_solution, "ey_")
            # add molrank
            g = add_molrank_properties(g)
            # add shorstest path
            g = add_shortest_path_properties(g)
            # save database
            save_to_sql(g, gid, output_dir)
            break

def safe_single_run(args):
    # try: 
    single_run(*args)
    # except:
        # pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Script to generate random graphs with given specifications")
    parser.add_argument("nspecies", type=int, help="List of species counts to simulate")
    parser.add_argument("nreactions", type=int, help="List of reaction counts to simulate")
    parser.add_argument("nrepeats", type=int, help="Number of repetitions for each configuration")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save the generated graphs")
    parser.add_argument("--timeout", type=float, default=300, help="Timeout for kinetic solver in seconds")
    parser.add_argument("--ntrails", type=int, default=10, help="Number of attempts to generate a fully connected network")
    parser.add_argument("--nworkers", type=int, default=16, help="Number of workers for execution")
    args = parser.parse_args()
    # build computation args
    comp_args = [
        (args.nspecies, args.nreactions, rep, args.timeout, args.output_dir, args.ntrails)
        for rep in range(args.nrepeats)
    ]
    # single_run(*comp_args[0])

    # run computaion
    with Pool(processes=args.nworkers) as pool:
        for _ in pool.imap_unordered(safe_single_run, comp_args):
            pass