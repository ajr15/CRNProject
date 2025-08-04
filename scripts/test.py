from src.NasaPolynomial import NasaPolynomial
import torinanet as tn
from typing import List, Set
from itertools import chain
import numpy as np
import networkx as nx
import json

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

def molrank2(rxn_graph: tn.core.RxnGraph, rate_constant_property: str, prefix: str, report_freq: int=50) -> tn.core.RxnGraph:
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
        if counter % report_freq == 0:
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
    # after finishing, update rxn_graph's objects
    for node in g:
        props = {prefix + k: v for k, v in g.nodes[node].items() if not k in "obj"}
        g.nodes[node]["obj"].properties.update(props)
    # return rxn graph
    return rxn_graph

def add_specie_enthalpy(rxn_graph: tn.core.RxnGraph):
    for sp in rxn_graph.species:
        if "thermo" in sp.properties:
            s = sp.properties["thermo"].replace("'", "\"")
            d = json.loads(s)
            poly = NasaPolynomial(**d)
            sp.properties["energy"] = poly.h(constants.DEFAULT_TEMPERATURE)
    return rxn_graph    

def add_reaction_energy(rxn_graph: tn.core.RxnGraph) -> tn.core.RxnGraph:
    rxn_graph = add_specie_enthalpy(rxn_graph)
    # adding data to reactions
    for rxn in rxn_graph.reactions:
        products_e = [s.properties.get("energy", None) for s in rxn.products]
        reactants_e = [s.properties.get("energy", None) for s in rxn.reactants]
        if None in products_e or None in reactants_e:
            continue
        rxn.properties["energy"]  = sum(products_e) - sum(reactants_e)
    return rxn_graph

def add_rate_constants(g: tn.core.RxnGraph):
    for rxn in g.reactions:
        A = rxn.properties["A"]
        beta = rxn.properties["beta"]
        Ea = rxn.properties["Ea"]
        T = constants.DEFAULT_TEMPERATURE
        rxn.properties["k"] = A * T ** beta * np.exp(- Ea / (T * constants.R))
        rxn.properties["ek"] = np.exp(- max(rxn.properties["energy"], 0) / (T * constants.R))
    return g

def add_kinetics(g: tn.core.RxnGraph, **kwargs):
    solver = tn.analyze.kinetics.KineticAnalyzer(g, "k")
    iconcs = [1 if s in g.source_species else 0 for s in g.species]
    solver.solve_kinetics(iconcs, **kwargs)
    for rxn in g.reactions:
        rxn.properties["max_rate"] = solver.get_max_rate(rxn)
    return g


if __name__ == "__main__":
    from src import constants
    from src.SimulatedReactionGraph import SimulatedReactionGraph
    import pandas as pd
    import matplotlib.pyplot as plt
    graph = "methane"
    path = constants.BASE_LITERATURE_GRAPH_PATH.replace("$", graph)
    rxn_graph = tn.core.RxnGraph.from_file(path, constants.RXN_GRAPH_PARAMS)
    # path = "data/simulated/{}/1.json".format(graph)
    # rxn_graph = SimulatedReactionGraph.from_file(path).to_rxngraph()
    rxn_graph = add_reaction_energy(rxn_graph)
    rxn_graph = add_rate_constants(rxn_graph)
    rxn_graph = molrank2(rxn_graph, "k", "real_")
    rxn_graph = molrank2(rxn_graph, "ek", "energy_")
    print("molrank done!")
    rxn_graph = add_kinetics(rxn_graph, **constants.KINETIC_SOLVER_KWARGS)
    data = []
    for rxn in rxn_graph.reactions:
        data.append([rxn.pretty_string(), rxn.properties["real_rate"], rxn.properties["energy_rate"], rxn.properties["max_rate"]])
    df = pd.DataFrame(data, columns=["rxn", "real_molrank", "energy_molrank", "max_rate"])
    df.to_csv("results/" + graph + ".csv")