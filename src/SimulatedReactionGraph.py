from dataclasses import dataclass
import networkx as nx
from typing import List, Optional
import json
import numpy as np
import torinanet as tn
from src import constants


class DummyAcMatrix (tn.core.AcMatrix):

    """this class is for cheating TorinaNet to accept dummy species without proper molecular definition (only symbol)"""

    def __init__(self, symbol: str=None):
        self.symbol = symbol

    def get_atom(self, i):
        """Abstract method to get the atomic number of the atom in the i-th index in AC matrix.
        ARGS:
            - i (int): index of the desired atom
        RETURNS:
            (int) atomic number of atom"""
        raise NotImplementedError

    def get_neighbors(self, i):
        """Abstract method to get the neighbors (as list of atom indecis) of atom in the i-th index in the AC matrix.
        ARGS:
            - i (int): index of the desired atom
        RETURNS:
            (list) list of atom indecis of the neighboring atoms"""
        raise NotImplementedError

    def to_networkx_graph(self):
        """Abstract method to convert AC matrix object to an undirected NetworkX graph"""
        raise NotImplementedError

    def from_networkx_graph(G):
        """Abstract method to read AC matrix from NetworkX object"""
        raise NotImplementedError

    def from_specie(specie):
        """Abastract method to create AC matrix from a Specie"""
        return DummyAcMatrix(specie.identifier)

    def to_specie(self):
        """Abastract method to convert an AC matrix to a Specie"""
        return tn.core.Specie(identifier=self.symbol)

    def get_compoenents(self):
        """Method to decompose an AC matrix to its components"""
        raise NotImplementedError
    
    def get_atoms(self):
        raise NotImplementedError

    def _to_str(self) -> str:
        """Method to write the ac matrix as a string (for internal use)"""
        return self.symbol

    def _from_str(self, string: str) -> None:
        """Method to read ac matrix from string (for internal use)"""
        self.__init__(string)

    def __len__(self):
        raise NotImplementedError

    def copy(self):
        return DummyAcMatrix(self.symbol)

    def build_geometry(self, connected_molecules: Optional[dict]=None):
        raise NotImplementedError("Cannot build geometry for the required AcMatrix object")

    def __eq__(self, other):
        return other.symbol == self.symbol


def dummy_reaction_hash(r: tn.core.Reaction) -> str:
    """Generate a hash for a reaction"""
    reactants = "+".join(sorted([s.identifier for s in r.reactants]))
    products = "+".join(sorted([s.identifier for s in r.products]))
    return "r:{}->{}".format(reactants, products)

def dummy_specie_hash(s: tn.core.Specie) -> str:
    return s.identifier

def hash_generator(base_func):
    while True:
        yield base_func, base_func

class DummyHashedCollection (tn.core.HashedCollection.CuckooHashedCollection):
    
    """Dummy hashed collection for dummy species and reactions."""

    def __init__(self, kind: str):
        self.kind = kind
        if kind == "species":
            super().__init__(hash_func_generator=hash_generator(dummy_specie_hash))
        elif kind == "reactions":
            super().__init__(hash_func_generator=hash_generator(dummy_reaction_hash))

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
        }

@dataclass
class GraphObject:

    oid: str
    properties: dict


class SimulatedReactionGraph:

    def __init__(self):
        self.g = nx.DiGraph()
        self.origin = None
        self.properties = {}

    def has_specie(self, idx: int) -> bool:
        return "s" + str(idx) in self.g
    
    def has_reaction(self, reactants: List[int], products: List[int]) -> bool:
        reactants = sorted(["s" + str(i) for i in reactants])
        products = sorted(["s" + str(i) for i in products])
        rxn = "r:{}->{}".format("+".join(reactants), "+".join(products))
        return rxn in self.g

    def add_specie(self, idx: int, energy: float, **properties): 
        """add a specie to the graph"""
        if self.has_specie(idx):
            raise ValueError("Specie with index {} is already in graph".format(idx))
        self.g.add_node("s" + str(idx), energy=energy, **properties)

    def add_reaction(self, reactants: List[int], products: List[int], **properties):
        """add a reaction between reactant specie idxs and product specie idxs"""
        # reformat reactants and species to node notation
        reactants = sorted(["s" + str(i) for i in reactants if i != 0])
        products = sorted(["s" + str(i) for i in products if i != 0])
        if self.has_reaction(reactants, products):
            raise ValueError("The reaction {} is already in graph".format("{}->{}".format("+".join(reactants), "+".join(products))))
        # calculate reaction energy
        if "energy" not in properties:
            r_energy = sum([self.g.nodes[s]["energy"] for s in products]) -\
                sum([self.g.nodes[s]["energy"] for s in reactants])
            properties["energy"] = r_energy
        rxn = "r:{}->{}".format("+".join(reactants), "+".join(products))
        # adding reaction to the graph
        self.g.add_node(rxn, **properties)
        for s in reactants:
            self.g.add_edge(s, rxn)
        for s in products:
            self.g.add_edge(rxn, s)

    def remove_reaction(self, reaction: str):
        """Remove a reaction given an ID from the graph"""
        if not reaction in self.g:
            raise ValueError("Specified reaction is not in the graph")
        created_sps = self.get_products(reaction) 
        # remove the reaction node from graph
        self.g.remove_node(reaction)
        # remove all orphand created species
        for sp in created_sps:
            if sp not in self.g:
                continue
            if len(self.creating_reactions(sp)) == 0:
                self.remove_specie(sp)

    def remove_specie(self, specie: str):
        if not specie in self.g:
            raise ValueError("Specified specie is not in the graph")
        # remove also all reactions of the specie
        rxns = self.consuming_reactions(specie) + self.creating_reactions(specie)
        # remove the specie and reaction nodes from graph
        self.g.remove_node(specie)
        for rxn in rxns:
            if rxn in self.g:
                self.remove_reaction(rxn)

    def reactions(self) -> List[GraphObject]:
        """Generator for all reactions (as GraphObject) in the graph"""
        res = []
        for node in self.g.nodes:
            if "r" in node:
                res.append(GraphObject(node, self.g.nodes[node]))
        return res
    
    def species(self) -> List[GraphObject]:
        """Iterator over all reactions (as GraphObject) in the graph"""
        res = []
        for node in self.g.nodes:
            if "r" not in node:
                res.append(GraphObject(node, self.g.nodes[node]))
        return res

    def consuming_reactions(self, specie: str) -> List[str]:
        """Get ids of consuming reactions of a specie"""
        return list(self.g.successors(specie))
    
    def creating_reactions(self, specie: str) -> List[str]:
        """Get ids of creating reactions of a specie"""
        return list(self.g.predecessors(specie))
    
    def get_reactants(self, reaction: str, include_null: bool=False) -> List[str]:
        """Get ids of reactant species of a reaction"""
        return [x for x in self.g.predecessors(reaction) if include_null or (not include_null and x != "s0")]

    def get_products(self, reaction: str, include_null: bool=False) -> List[str]:
        """Get ids of product species of a reaction"""
        return [x for x in self.g.successors(reaction) if include_null or (not include_null and x != "s0")]
    
    def get_properties(self, node: str) -> dict:
        """Get properties dictionary of a node (reaction or specie)"""
        if node in self.g:
            return self.g.nodes[node]
        else:
            raise ValueError("Node id {} is not in reaction graph".format(node))
        
    def get_species(self) -> List[str]:
        """Get all specie node ids"""
        return [s for s in self.g.nodes if "->" not in s and not s == "s0"]
    
    def get_reactions(self) -> List[str]:
        """Get all specie node ids"""
        return [s for s in self.g.nodes if "r" in s]
    
    @staticmethod
    def generate_activation_energy(rxn_energy: float, **dist_kwargs):
        kwds = dist_kwargs.copy()
        kwds.update(constants.EA_DIST_ARGUMENTS)
        if rxn_energy < 0:
            return kwds["E<0"].rvs()
        else:
            base_value = kwds["a"] * rxn_energy + kwds["b"] # prediction from fitted values
            error = kwds["error"].rvs() # error from fitted values
            return base_value + error
    
    def add_kinetic_data(self, preexponential_property_name: str="A", beta_property_name: str="beta", activation_energy_property_name: str="Ea", logk_property_name: str="logk", **distribution_kwargs):
        """Add kinetic rate constant information of the networks. the estimate is based on eyring polanyi equations (with reaction energy) and optionally, can add a 'real' constant by adding a differnce distribution.
        Using the distribution kwargs, one can control the distribution of the different sizes. the values are {"A": A dist, "beta": beta dist, "Ea": {"T": temperature for logk calculation, "E<0": dist when E < 0, "a": slope of Ea vs E, "b": intercept of Ea vs E, "error": error distribution of Ea vs E}}"""
        kw = distribution_kwargs.copy()
        if "A" not in kw:
            kw["A"] = constants.PREEXPONTIAL_DIST
        if "beta" not in kw:
            kw["beta"] = constants.BETA_DIST
        if "Ea" not in kw:
            kw["Ea"] = constants.EA_DIST_ARGUMENTS
        if "T" not in kw:
            kw["T"] = constants.DEFAULT_TEMPERATURE
        # define generating distributions for A and beta, they are fitted to real data.
        A = kw["A"]
        beta = kw["beta"]
        T = kw["T"]
        for rxn in self.reactions():
            sampled_A = A.rvs()
            sampled_Ea = self.generate_activation_energy(rxn.properties["energy"], **kw["Ea"])
            sampled_beta = beta.rvs()
            rxn.properties[activation_energy_property_name] = sampled_Ea
            rxn.properties[preexponential_property_name] = sampled_A
            rxn.properties[beta_property_name] = sampled_beta
            rxn.properties[logk_property_name] = np.log(sampled_A) + sampled_beta * np.log(T) - sampled_Ea / (constants.R * T)

    def set_origin(self, origin: Optional[str]=None):
        """Optionally set an origin to the graph. if not explicitly chosen, it is set randomly"""
        if origin is None:
            origin = np.random.choice(self.get_species())
        self.origin = origin

    def copy(self):
        ajr = SimulatedReactionGraph()
        ajr.g = self.g.copy()
        ajr.set_origin(self.origin)
        return ajr
    
    def save(self, path: str):
        d = {
            "species": {},
            "reactions": {},
            "origin": self.origin,
            "properties": self.properties
        } 
        for s in self.species():
            d["species"][s.oid] = s.properties
        for r in self.reactions():
            d["reactions"][r.oid] = r.properties
        with open(path, "w") as f:
            json.dump(d, f, indent=4)

    @staticmethod
    def from_file(path: str) -> 'SimulatedReactionGraph':
        with open(path, "r") as f:
            data = json.load(f)
        g = SimulatedReactionGraph()
        for s, props in data["species"].items():
            energy = props.pop("energy")
            sidx = int(s[1:])
            g.add_specie(sidx, energy=energy, **props)
        for r, props in data["reactions"].items():
            r = r[2:]
            reactants = [int(x[1:]) for x in r.split("->")[0].split("+")]
            products = [int(x[1:]) for x in r.split("->")[1].split("+")]
            g.add_reaction(reactants, products, **props)
        g.origin = data.get("origin", None)
        g.properties = data["properties"]
        return g

    def to_rxngraph(self) -> tn.core.RxnGraph:
        rxn_graph = tn.core.RxnGraph(ac_matrix_type=DummyAcMatrix,
                                 reaction_collection=DummyHashedCollection, 
                                 specie_collection=DummyHashedCollection,
                                 specie_collection_kwargs={"kind": "species"},
                                 reaction_collection_kwargs={"kind": "reactions"})
        # add species to graph
        for sp in self.species():
            if sp.oid != "s0":
                ajr = tn.core.Specie(sp.oid, sp.properties)
                rxn_graph.add_specie(ajr)
        # add reactions
        for rxn in self.reactions():
            reactants = [tn.core.Specie(x, self.get_properties(x)) for x in self.g.predecessors(rxn.oid) if x != "s0"]
            products = [tn.core.Specie(x, self.get_properties(x)) for x in self.g.successors(rxn.oid) if x != "s0"]
            ajr = tn.core.Reaction(reactants, products, rxn.properties)
            rxn_graph.add_reaction(ajr)
        # set source species
        if self.origin is not None:
            og = [self.origin] if type(self.origin) is str else self.origin
            source = [tn.core.Specie(x, self.get_properties(x)) for x in og]
            rxn_graph.set_source_species(source)
        rxn_graph.properties = self.properties
        return rxn_graph
    
if __name__ == "__main__":
    # test the rxngraph conversion
    g = SimulatedReactionGraph.from_file("data/simulated/s25r1000/28.json")
    g.origin = np.random.choice(g.get_species(), 2)
    rxn_graph = g.to_rxngraph()
    reducer = tn.analyze.network_reduction.KineticReduction.MolRankReduction(0, "k", estimate_max_constants=False)
    analyzer = tn.analyze.algorithms.ShortestPathAnalyzer(rxn_graph, prop_func=lambda rxn: 1)
    print(analyzer.shortest_path_table)
    molrank_reactions_df = reducer.rank_reactions(rxn_graph)
    print(molrank_reactions_df)
    molrank_species_df = reducer.rank_species(rxn_graph)
    print(molrank_species_df)
