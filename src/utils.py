import torinanet as tn
import numpy as np
import signal
from typing import Set, List
from itertools import chain
import networkx as nx
import pandas as pd
from src import constants
from src.GeneratingModels import PreferentialAttachmentModel


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Kinetics solving timed out")


signal.signal(signal.SIGALRM, timeout_handler)


def solve_kinetics(g: tn.core.RxnGraph, timeout: float, temperature: float, rate_constant_property: str, **solver_kwargs):
    signal.alarm(timeout)
    try:
        g = add_rate_constants(g, temperature)
        solver = tn.analyze.kinetics.KineticAnalyzer(g, rate_constant_property)
        iconcs = [1 if s in g.source_species else 0 for s in g.species]
        solver.solve_kinetics(iconcs, **solver_kwargs)
        signal.alarm(0)  # Cancel the alarm if successful
        return solver
    except TimeoutException:
        return None


def add_rate_constants(g: tn.core.RxnGraph, temp: float):
    for r in g.reactions:
        props = r.properties
        # Calculate the rate constant using the Arrhenius equation
        r.properties["k"] = props["A"] * np.power(temp, props["beta"]) * np.exp(- props["Ea"] / (constants.R * temp))
        # add also "eyring polanyi" approximation of the rate constant - 
        # with slight twist, all positive reactions are set to have same constant
        # we use the energy to Ea realationship fitted for network generation.
        estimated_ea = constants.EA_DIST_ARGUMENTS["a"] * props["energy"] + constants.EA_DIST_ARGUMENTS["b"] if props["energy"] > 0 else np.exp(constants.EA_DIST_ARGUMENTS["E<0"].mean)
        r.properties["k_ey"] = temp * constants.BOLTZMANN / constants.PLANK * np.exp(- estimated_ea / (constants.R * temp))
    return g


def generate_network(n_species, n_reactions, sp_energy_dist) -> tn.core.RxnGraph:
    model = PreferentialAttachmentModel(n_species=n_species, 
                                        n_reactions=n_reactions, 
                                        specie_energy_dist=sp_energy_dist)
    g = model.generate_graph()
    g.origin = list(np.random.choice(g.get_species(), 2)) # set randomly two origin points
    g.add_kinetic_data()
    return g.to_rxngraph()


def consuming_reactions(g: nx.DiGraph, species: List[str], exclusive: bool) -> Set[str]:
    ajr = set()
    for suc in chain(*[g.successors(node) for node in species]):
        if exclusive and all([p in species for p in g.predecessors(suc)]):
            ajr.add(suc)
        elif not exclusive:
            ajr.add(suc)
    return ajr


def creating_reactions(g: nx.DiGraph, species: List[str]) -> Set[str]:
    return set(chain(*[g.predecessors(s) for s in species]))


def reactions_products(g: nx.DiGraph, reactions: List[str]) -> Set[str]:
    return set(chain(*[g.successors(r) for r in reactions]))


def reactions_reactants(g: nx.DiGraph, reactions: List[str]) -> Set[str]:
    return set(chain(*[g.predecessors(r) for r in reactions]))


def molrank_ranking(rxn_graph: tn.core.RxnGraph, rate_constant_property: str, verbose: int=0, report_freq: int=50) -> nx.DiGraph:
    """Run a MolRank analysis of the network, returns a nx.DiGraph with the newtork proeprties"""
    g = rxn_graph.to_networkx_graph(use_internal_id=True)
    # initialize all reactions to "unvisited" with rate and flux 0
    all_reactions = [rxn_graph.reaction_collection.get_key(rxn) for rxn in rxn_graph.reactions]
    for rxn in all_reactions:
        g.nodes[rxn]["visited"] = False
        g.nodes[rxn]["rate"] = 0
        g.nodes[rxn]["flux"] = 0
        # add also rate constant to be accessible from graph
        g.nodes[rxn][rate_constant_property] = g.nodes[rxn]["obj"].properties[rate_constant_property]
    # set initial reactants - the graph source
    reactants = set([rxn_graph.specie_collection.get_key(s) for s in rxn_graph.source_species])
    # initialize all specie times and concentrations to 0
    all_species = [rxn_graph.specie_collection.get_key(sp) for sp in rxn_graph.species]
    for specie in all_species:
        g.nodes[specie]["time"] = 0
        g.nodes[specie]["conc"] = 0
        # source species get a conc of 1
        if specie in reactants:
            g.nodes[specie]["conc"] = 1
    counter = 0
    while True:
        counter += 1
        # get all consuming reactions of the reactants set
        reactions = consuming_reactions(g, reactants, exclusive=True)
        # filter out visited reactions
        reactions = [r for r in reactions if not g.nodes[r]["visited"]]
        # if reactions is an empty list - we are done, all reactions are visited
        if counter % report_freq == 0 and verbose > 0:
            print("iteration", counter, "# reactions", len(reactions), "# reactants", len(reactants))
        if len(reactions) == 0:
            break
        # set reaction rates
        for r in reactions:
            g.nodes[r]["rate"] = g.nodes[r][rate_constant_property] * np.prod([g.nodes[s]["conc"] for s in reactions_reactants(g, [r])])
            # once the rate is set - reaction is visited :)
            g.nodes[r]["visited"] = True
        # set specie "times"
        for s in reactants:
            ajr = [r for r in consuming_reactions(g, [s], exclusive=False) if r in reactions] # get ALL consuming reactions of specie in current step - crucial
            g.nodes[s]["time"] = g.nodes[s]["conc"] / np.sum([g.nodes[r]["rate"] for r in ajr])
        # calcuate reaction fluxes
        for r in reactions:
            g.nodes[r]["flux"] = g.nodes[r]["rate"] * np.min([g.nodes[s]["time"] for s in reactions_reactants(g, [r])])
        # find all reaction products
        products = reactions_products(g, reactions)
        # set products conc based on reaction flux
        for p in products:
            conc = np.sum([g.nodes[r]["flux"] for r in creating_reactions(g, [p])])
            # if conc is bigger than the exising one, set all consuming reactions to unvisited - their rate might not be maximal
            if conc > g.nodes[p]["conc"]:
                for r in consuming_reactions(g, [p], exclusive=False):
                    g.nodes[r]["visited"] = False
                # now, just set the conc - ONLY IF ITS LARGER !
                g.nodes[p]["conc"] = conc
        # add products to reactants set and continue
        reactants = reactants.union(products)
    return g

