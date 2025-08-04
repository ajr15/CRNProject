from src.GeneratingModels import SimulatedReactionGraph
from src.KineticSolver import KineticSolver
from src.NasaPolynomial import NasaPolynomial
import torinanet as tn
from scipy.interpolate import interp1d
import numpy as np
from typing import Union, List
import networkx as nx
import json
import matplotlib.pyplot as plt
import pandas as pd

def solve_kinetics(g: Union[tn.core.RxnGraph, SimulatedReactionGraph], temperature: float, **solver_kwargs):
    if isinstance(g, SimulatedReactionGraph):
        solver = KineticSolver(g, "k")
        og = np.random.choice(g.get_species(), 2) # set randomly two origin points
        iconcs = [1 if s in s in og else 0 for s in g.get_species()]
        reactions = g.reactions()
    elif isinstance(g, tn.core.RxnGraph):
        solver = tn.analyze.kinetics.KineticAnalyzer(g, "k")
        iconcs = [1 if sp in g.source_species else 0 for sp in g.species]
        reactions = g.reactions
    # add rate constant property
    for rxn in reactions:
        rxn.properties["k"] = rxn.properties["A"] * temperature ** rxn.properties["beta"] * np.exp(- rxn.properties["Ea"] / (temperature * 8.314))
    # solve kinetics
    solver.solve_kinetics(iconcs, **solver_kwargs)
    return solver

def compare_concentrations(osolver: Union[tn.analyze.kinetics.KineticAnalyzer, KineticSolver], rsolver: Union[tn.analyze.kinetics.KineticAnalyzer, KineticSolver]) -> dict:
    """Compare different kinetic solveres of original (o) and reduced/percolated (r) graphs"""
    odf = osolver.concentrations_df()
    rdf = rsolver.concentrations_df()
    # compare concentration values - profile + ss value
    profile_err = 0
    ss_err = 0
    for sp in rdf.columns:
        f = interp1d(rdf.index, rdf[sp], bounds_error=False)
        profile_err += np.sum(np.abs(odf[sp] - f(odf.index)))
        oss = odf[sp].iloc[-1]
        rss = rdf[sp].iloc[-1]
        ss_err += np.abs(oss - rss)
    # calculate average
    nsps = len(rdf.columns)
    profile_err = profile_err / (nsps * len(odf.index))
    ss_err = ss_err / nsps
    # normalize by average conc values
    return {"conc_profile_error": profile_err, "conc_ss_error": ss_err}

def compare_rates(osolver: Union[tn.analyze.kinetics.KineticAnalyzer, KineticSolver], rsolver: Union[tn.analyze.kinetics.KineticAnalyzer, KineticSolver]) -> dict:
    odf = osolver.rates_df()
    rdf = rsolver.rates_df()
    # compare concentration values - profile + ss value
    profile_err = 0
    ss_err = 0
    for rxn in rdf.columns:
        f = interp1d(rdf.index, rdf[rxn], bounds_error=False)
        profile_err += np.sum(np.abs(odf[rxn] - f(odf.index)))
        oss = odf[rxn].max()
        rss = rdf[rxn].max()
        ss_err += np.abs(oss - rss)
    # calculate average
    nsps = len(rdf.columns)
    profile_err = profile_err / (nsps * len(odf.index))
    ss_err = ss_err / nsps
    # normalize by average conc values
    return {"rate_profile_error": profile_err, "rate_max_error": ss_err}


def covered_rate(ogsolver: Union[tn.analyze.kinetics.KineticAnalyzer, KineticSolver], rg: Union[tn.core.RxnGraph, SimulatedReactionGraph]):
    if isinstance(rg, SimulatedReactionGraph):
        reactions = rg.get_reactions()
    elif isinstance(rg, tn.core.RxnGraph):
        reactions = [rg.reaction_collection.get_key(r) for r in rg.reactions]
    rates = ogsolver.rates_df()
    covered = rates.loc[:, reactions].sum(axis=1).values / rates.sum(axis=1).values
    return {"min_coverage": np.min(covered), "mean_coverage": np.mean(covered), "ss_coverage": covered[-1]}


class ReactionEnergyReduction:

    def __init__(self, th: float, leaf_max_in_degree: int=0, leaf_energy_th: float=np.inf):
        self.leaf_max_in_degree = leaf_max_in_degree
        self.leaf_energy_th = leaf_energy_th
        self.th = th

    @staticmethod
    def specie_enthalpy(sp: tn.core.Specie, temperature: float) -> float:
        if not "thermo" in sp.properties:
            return None
        s = sp.properties["thermo"].replace("'", "\"")
        d = json.loads(s)
        poly = NasaPolynomial(**d)
        return poly.h(temperature)

    @classmethod
    def calculate_reaction_energy(cls, rxn: tn.core.Reaction) -> float:
        return sum([cls.specie_enthalpy(sp, temperature=600) for sp in rxn.products]) - sum([cls.specie_enthalpy(sp, temperature=600) for sp in rxn.reactants])

    def is_leaf(self, G: nx.DiGraph, node: str) -> bool:
        products = G.successors(node)
        return any([len(G.predecessors(s)) <= self.leaf_max_in_degree for s in products])

    def apply_simulated(self, g: SimulatedReactionGraph) -> SimulatedReactionGraph:
        for rxn in g.reactions():
            if rxn.oid in g.g and rxn.properties["energy"] > self.th:
                g.remove_reaction(rxn.oid)
            elif self.is_leaf(g.g, rxn.oid) and rxn.properties["energy"] > self.leaf_energy_th:
                g.remove_reaction(rxn.oid)
        return g
    
    def apply_real(self, g: tn.core.RxnGraph) -> tn.core.RxnGraph:
        rxns = list(g.reactions)
        G = g.to_networkx_graph(use_internal_id=True)
        for rxn in rxns:
            rid = g.reaction_collection.get_key(rxn)
            if not "energy" in rxn.properties:
                rxn.properties["energy"] = self.calculate_reaction_energy(rxn)
            if g.has_reaction(rxn) and rxn.properties["energy"] > self.th:
                g = g.remove_reaction(rxn)
            elif self.is_leaf(G, rid) and rxn.properties["energy"] > self.leaf_energy_th:
                g.remove_reaction(rxn.oid)
        return g
    
    def to_dict(self) -> dict:
        return {
            "type": "energy_reduction",
            "th": self.th,
            "leaf_max_in_degree": self.leaf_max_in_degree,
            "leaf_energy_th": self.leaf_energy_th
        }

