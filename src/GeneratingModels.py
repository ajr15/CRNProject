from scipy.stats import rv_continuous, binom
from math import comb
import numpy as np
from typing import Optional
import networkx as nx
from src.SimulatedReactionGraph import SimulatedReactionGraph
from src.Distribution import Distribution

class PreferentialAttachmentModel:


    def __init__(self, n_species: int, n_reactions: int, specie_energy_dist: rv_continuous, alpha: float=1) -> None:
        self.n_species = n_species
        self.n_reactions = n_reactions
        self.alpha = alpha
        self.specie_energy_dist = specie_energy_dist
        if comb(n_species + 1, 4) < n_reactions:
            raise ValueError("Cannot create a network with {} species and {} reactions. try increasing number of species or reduce number of reactions")

    def sample_energy(self) -> float:
        return self.specie_energy_dist.rvs()

    def generate_graph(self) -> SimulatedReactionGraph:
        g = SimulatedReactionGraph()
        # add species 
        g.add_specie(0, energy=0) # dummy specie
        for i in range(self.n_species):
            g.add_specie(i + 1, energy=self.sample_energy())
        # sets origin
        g.set_origin()
        # adding reactions
        rxncounter = 0
        while rxncounter < self.n_reactions:
            # get a species list
            species = list(range(self.n_species + 1))
            # calcualte "out degree" parameters for all species - based on them we determine the choice probabilities
            degrees = [1 + len(g.consuming_reactions("s" + str(s))) ** self.alpha for s in species]
            probabilities = [d / sum(degrees) for d in degrees]
            # select reactants (with preferential probabilities)
            reactants = np.random.choice(species, replace=False, size=2, p=probabilities)
            # remove them from available species (unless they are the dummy specie)
            species = [s for s in species if s not in reactants or s == 0]
            # calculate "in degree" parameters for all remaining species - based on them we determine the choice probabilities of the products
            degrees = [1 + len(g.creating_reactions("s" + str(s))) ** self.alpha for s in species]
            probabilities = [d / sum(degrees) for d in degrees]
            # choose products out of the remaining species
            products = np.random.choice(species, replace=False, size=2, p=probabilities)
            if not g.has_reaction(reactants, products):
                g.add_reaction(reactants, products)
                rxncounter += 1
        return g


class RandomGraphModel:

    def __init__(self, n_species: int, n_reactions: int, specie_energy_dist: Optional[rv_continuous], max_energy: Optional[float]=None, n_sample_points: int=100000, energy_isf_th: float=1e-4) -> None:
        self.n_species = n_species
        self.n_reactions = n_reactions
        self.max_energy = max_energy
        self.specie_energy_dist = specie_energy_dist
        self.n_sample_points = n_sample_points
        self.energy_isf_th = energy_isf_th
        if comb(n_species + 1, 4) < n_reactions:
            raise ValueError("Cannot create a network with {} species and {} reactions. try increasing number of species or reduce number of reactions")

    def generate_graph(self):
        g = nx.DiGraph()
        # add species 
        g.add_node("s0", energy=0) # dummy specie
        for i in range(self.n_species):
            g.add_node("s{}".format(i + 1), energy=self.specie_energy_dist.rvs())
        # adding reactions
        rxncounter = 0
        rxns = set()
        while rxncounter < self.n_reactions:
            # get a species list
            species = list(range(self.n_species + 1))
            # select reactants
            reactants = np.random.choice(species, replace=False, size=2)
            # remove them from available species (unless they are the dummy specie)
            species = [s for s in species if s not in reactants or s == 0]
            # choose products out of the remaining species
            products = np.random.choice(species, replace=False, size=2)
            # make rxnstring to ensure unique reactions
            s = ",".join([str(x) for x in sorted(reactants) + sorted(products)])
            if s in rxns:
                continue
            rxns.add(s)
            # reformat reactants and species to node notation
            reactants = ["s" + str(i) for i in reactants]
            products = ["s" + str(i) for i in products]
            # calculate reaction energy
            r_energy = sum([g.nodes[s]["energy"] for s in products]) - sum([g.nodes[s]["energy"] for s in reactants])
            # if required, filter by energy
            if self.max_energy is not None and r_energy > self.max_energy:
                continue
            # adding reaction to the graph
            rxn = "r{}".format(rxncounter + 1)
            g.add_node(rxn, energy=r_energy)
            for s in reactants:
                g.add_edge(s, rxn)
            for s in products:
                g.add_edge(rxn, s)
            rxncounter += 1
        return g

    def reaction_energy_distribution(self, nreactants: int, nproducts: int) -> Distribution:
        """Get the expected energy distribution of reaction with nreactants and nproducts"""
        ajr = Distribution.from_cont_dist(self.specie_energy_dist, self.n_sample_points, self.energy_isf_th)
        dist = ajr
        for _ in range(nproducts - 1):
            dist += ajr
        for _ in range(nreactants):
            dist -= ajr
        return dist
    
    def total_energy_distribution(self) -> Distribution:
        r2p2 = self.reaction_energy_distribution(2, 2)
        r1p2 = self.reaction_energy_distribution(1, 2)
        r2p1 = self.reaction_energy_distribution(2, 1)
        r1p1 = self.reaction_energy_distribution(1, 1)
        N = self.n_species
        return r1p1 * (2 / (N * (N + 1))) + r2p1 * (2 / (N + 1)) + r1p2 * 2 * ((N - 2) / (N * (N + 1))) + r2p2 * ((N - 3) / (N + 1))
    
    def log_stirling_factorial(self, n: int):
        return np.log(np.sqrt(2 * np.pi * n)) + n * np.log(n / np.e)

    def degree_distribution(self, degree: int, p: float):
        """analytic degree distribution calculator"""
        return binom(self.n_reactions, p).pmf(degree)

    def total_degree(self, degree: int) -> float:
        p = 4 / (self.n_species + 1)
        return self.degree_distribution(degree, p)

    def in_degree(self, degree: int) -> float:
        p = 2 / (self.n_species + 1)
        return self.degree_distribution(degree, p)

    def out_degree(self, degree: int) -> float:
        p = 2 / (self.n_species + 1)
        return self.degree_distribution(degree, p)
       