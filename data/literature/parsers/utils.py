from typing import List, Optional
from dataclasses import dataclass
from copy import copy
import sqlite3
import openbabel as ob
import json
import torinanet as tn


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



@dataclass
class DummySpecie:

    symbol: str
    charge: int
    properties: dict

    def __str__(self):
        return self.symbol

    def __hash__(self) -> int:
        return hash(self.symbol)

    def to_specie(self, symbol_dict: Optional[dict]=None) -> tn.core.Specie:
        props = copy(self.properties)
        props["symbol"] = self.symbol
        s = self.symbol
        if symbol_dict is not None:
            s = symbol_dict[s]
        sp = tn.core.Specie(s, charge=self.charge, properties=props)
        # must use a dummy ac matrixs for using TorinaNet
        sp.ac_matrix = DummyAcMatrix(self.symbol)
        return sp


@dataclass
class DummyReaction:

    reactants: List[str]
    products: List[str]
    properties: dict

    def __hash__(self):
        return hash("{}->{}".format("+".join(self.reactants), "+".join(self.products)))
    
    def __repr__(self):
        return "DummyReaction(\"{}->{}\")".format("+".join(self.reactants), "+".join(self.products))
    
    def __eq__(self, other):
        if not isinstance(other, DummyReaction):
            return False
        return self.__hash__() == other.__hash__()


def build_rxn_tree(reactions: List[DummyReaction], source_species: List[str]):
    """Build the total reaction graph with the desired source species"""
    allowed_species = set(source_species)
    res = set()
    pcount = 0
    while True:
        count = 0
        new_species = set()
        for reaction in reactions:
            if all([r in allowed_species for r in reaction.reactants]):
                res.add(reaction)
                count += 1
                for r in reaction.products:
                    new_species.add(r)
        allowed_species = allowed_species.union(new_species)
        if count == pcount:
            break
        pcount = count
    return list(res)

def get_species_list(rxn_list: List[DummyReaction]):
    species = set()
    for r in rxn_list:
        for s in r.reactants:
            species.add(s)
        for s in r.products:
            species.add(s)
    return list(species)

def init_species_file(specie_list: List[str], specie_file: str):
    with open(specie_file, "w") as f:
        d = {s: "" for s in specie_list}
        json.dump(d, f, sort_keys=True, indent=1)

def dummy_reaction_hash(r: tn.core.Reaction) -> str:
    """Generate a hash for a reaction"""
    reactants = "+".join(sorted([s.identifier for s in r.reactants]))
    products = "+".join(sorted([s.identifier for s in r.products]))
    return "{}->{}".format(reactants, products)

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

def dummy_reactions_to_graph(rxn_list: List[DummyReaction], species: List[DummySpecie], source_species: List[str], specie_symbols: dict=None) -> tn.core.RxnGraph:
    # read species and source species
    species_dict = {s.symbol: s.to_specie(specie_symbols) for s in species}
    source_species = [species_dict[s] for s in source_species]
    # initialize rxn graph
    rxn_graph = tn.core.RxnGraph(ac_matrix_type=DummyAcMatrix,
                                 reaction_collection=DummyHashedCollection, 
                                 specie_collection=DummyHashedCollection,
                                 specie_collection_kwargs={"kind": "species"},
                                 reaction_collection_kwargs={"kind": "reactions"})
    source_species = [rxn_graph.add_specie(s) for s in source_species]
    rxn_graph.set_source_species(source_species, force=True)
    # read reactions
    covered_rxns = set()
    for rxn in rxn_list:
        if rxn in covered_rxns:
            continue
        # check if reaction species are in the species dict
        if not all([s in species_dict for s in rxn.reactants]) or not all([s in species_dict for s in rxn.products]):
            continue
        covered_rxns.add(rxn)
        reactants = [species_dict[s] for s in rxn.reactants]
        products = [species_dict[s] for s in rxn.products]
        reaction = tn.core.Reaction(reactants, 
                                    products, 
                                    properties=rxn.properties)
        rxn_graph.add_reaction(reaction)
    return rxn_graph

def get_energy(connection, smiles):
    l = connection.execute("SELECT energy FROM species WHERE smiles=\"{}\"".format(smiles)).fetchall()
    if len(l) > 0:
        return l[0][0]
    else:
        return None

def get_atomization_energy(connection, smiles: str, atom_energies: dict):
    if smiles is None:
        return None
    # read smiles
    conv = ob.OBConversion()
    conv.SetInFormat("smi")
    obmol = ob.OBMol()
    conv.ReadString(obmol, smiles)
    obmol.AddHydrogens()
    # get single atom energy
    ajr = 0
    for atom in ob.OBMolAtomIter(obmol):
        ajr += atom_energies[ob.GetSymbol(atom.GetAtomicNum())]
    e = get_energy(connection, smiles)
    if e is not None:
        return e - ajr
    else:
        return None

def add_atomization_energies(db_path, rxn_graph, symbol_to_smiles) -> tn.core.RxnGraph:
    """Method to update the species energies in the reaction graph from the computation"""
    connection = sqlite3.connect(db_path)
    atom_energies = {
                     "H": get_energy(connection, "[H]"), 
                     "O": get_energy(connection, "[O]"), 
                     "C": get_energy(connection, "[C]"),
                     "N": get_energy(connection, "[N]")
                    }
    for specie in rxn_graph.species:
        if specie.identifier in symbol_to_smiles:
            smiles = symbol_to_smiles[specie.identifier]
            atomization_energy = get_atomization_energy(connection, smiles, atom_energies)
            if atomization_energy is not None:
                specie.properties["atomization_energy"] = atomization_energy
    return rxn_graph
