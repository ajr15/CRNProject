import json
from dataclasses import dataclass
import pandas as pd
from scipy.stats.distributions import rv_continuous, rv_discrete, binom, norm
from scipy.signal import fftconvolve
from scipy.integrate import ode
from math import comb
import networkx as nx
import numpy as np
from typing import Optional, List

class LogNormal:

    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std
        self.dist = norm(mean, std)

    def rvs(self):
        return np.exp(self.dist.rvs())

class Distribution:

    def __init__(self, points, pdf):
        self.points = points
        self.pdf = pdf / np.trapz(pdf, points)

    def from_cont_dist(base_dist: rv_continuous, npoints: int=10000, th: float=1e-6):
        base_dist = base_dist
        max_v = base_dist.isf(th)
        X = np.linspace(-max_v, max_v, npoints)
        pdf = base_dist.pdf(X)
        return Distribution(X, pdf)
    
    def from_discrete_dist(base_dist: rv_discrete, min_val: int, max_val: int, *args):
        base_dist = base_dist
        X = np.array(range(min_val, max_val + 1))
        pdf = base_dist.pmf(X, *args)
        return Distribution(X, pdf)
    
    @staticmethod
    def add(dist1, dist2):
        """Computes the PDF of the sum of two distributions via convolution"""
        dx = dist1.points[1] - dist1.points[0]
        xmin = dist1.points[0] + dist2.points[0]
        xmax = dist1.points[-1] + dist2.points[-1]
        pdf_result = fftconvolve(dist1.pdf, dist2.pdf, mode='full') * dx
        new_x = np.linspace(xmin, xmax, len(pdf_result))
        return Distribution(new_x, pdf_result)
    
    @classmethod
    def substract(cls, dist1, dist2):
        """Compute the pdf of the difference of two distributions via convolution"""
        dx = dist1.points[1] - dist1.points[0]
        xmin = dist1.points[0] - dist2.points[-1]
        xmax = dist1.points[-1] - dist2.points[0]
        # we flip the PDF to use convolution also for substraction
        pdf_result = fftconvolve(dist1.pdf, dist2.pdf[::-1], mode='full') * dx
        new_x = np.linspace(xmin, xmax, len(pdf_result))
        return Distribution(new_x, pdf_result)
    
    def cdf(self, v: float) -> float:
        """calculate the CDF of the distribution"""
        # find index of closest point
        i = np.argmin([abs(a - v) for a in self.points])
        # calculate integral up to the closest point
        return np.trapz(self.pdf[:i], self.points[:i])
    
    def icdf(self, q: float) -> float:
        """Return the inverse CDF of the distribution"""
        for i, p in enumerate(self.points):
            cdf = np.trapz(self.pdf[:i], self.points[:i])
            if cdf >= q:
                return p
        return p
    
    def ppf(self, v: float) -> float:
        """calculate the PPF (equivalent to PDF) of the distribution"""
        # find index of closest point
        i = np.argmin([abs(a - v) for a in self.points])
        # calculate integral up to the closest point
        return self.pdf[i]
    
    def __add__(self, other):
        if not isinstance(other, Distribution):
            raise ValueError("Can only add distribution to another distribution")
        return self.add(self, other)
    
    def __sub__(self, other):
        if not isinstance(other, Distribution):
            raise ValueError("Can only substract distribution to another distribution")
        return self.substract(self, other)
    
    def __mul__(self, other):
        if not (type(other) is float or type(other) is int):
            raise ValueError("Can only multiply distribution by a scalar value")
        new_x = self.points * other
        new_pdf = self.pdf / abs(other)
        return Distribution(new_x, new_pdf)


@dataclass
class GraphObject:

    oid: str
    properties: dict


class SimulatedReactionGraph:

    def __init__(self):
        self.g = nx.DiGraph()
        self.origin = None

    def has_specie(self, idx: int) -> bool:
        return "s" + str(idx) in self.g
    
    def has_reaction(self, reactants: List[int], products: List[int]) -> bool:
        reactants = ["s" + str(i) for i in sorted(reactants)]
        products = ["s" + str(i) for i in sorted(products)]
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
        reactants = ["s" + str(i) for i in sorted(reactants)]
        products = ["s" + str(i) for i in sorted(products)]
        if self.has_reaction(reactants, products):
            raise ValueError("The reaction {} is already in graph".format("{}->{}".format("+".join(reactants), "+".join(products))))
        # calculate reaction energy
        r_energy = sum([self.g.nodes[s]["energy"] for s in products]) -\
              sum([self.g.nodes[s]["energy"] for s in reactants])
        rxn = "r:{}->{}".format("+".join(reactants), "+".join(products))
        # adding reaction to the graph
        self.g.add_node(rxn, energy=r_energy, **properties)
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
    def generate_activation_energy(rxn_energy: float):
        if rxn_energy < 0:
            return LogNormal(9.23, 1.40).rvs()
        else:
            base_value = 9.54e-1 * rxn_energy + 2.02e4 # prediction from fitted values
            error = norm(0, 3.55e4).rvs() # error from fitted values
            return base_value + error
    
    def add_kinetic_data(self, preexponential_property_name: str="A", beta_property_name: str="beta", activation_energy_property_name: str="Ea"):
        """Add kinetic rate constant information of the networks. the estimate is based on eyring polanyi equations (with reaction energy) and optionally, can add a 'real' constant by adding a differnce distribution"""
        # define generating distributions for A and beta, they are fitted to real data.
        A = LogNormal(28.66, 11.62)
        beta = norm(0.32, 1.23)
        for rxn in self.reactions():
            rxn.properties[activation_energy_property_name] = self.generate_activation_energy(rxn.properties["energy"])
            rxn.properties[preexponential_property_name] = A.rvs()
            rxn.properties[beta_property_name] = beta.rvs()

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
            "origin": self.origin
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
        return g

        

class KineticSolver:

    """Kinetic solver for a generated reaction graph"""

    def __init__(self, g: SimulatedReactionGraph, rate_constant_property_name: str="k"):
        self.g = g
        self.rate_constant_property_name = rate_constant_property_name
        self._specie_idx = {sp: i for i, sp in enumerate(g.get_species())}
        self._concs = []
        self._ts = []

    @staticmethod
    def _reaction_rate_function(t, concs, k, ridxs, pidxs):
        diff = np.zeros(len(concs))
        r = k * np.prod([concs[i] for i in ridxs])
        for i in ridxs:
            diff[i] -= r
        for i in pidxs:
            diff[i] += r
        return diff

    def _build_target_f(self):
        """build the function for the solver"""
        ks = []
        rids = []
        pids = []
        for rxn in self.g.get_reactions():
            rids.append([self._specie_idx[s] for s in self.g.get_reactants(rxn)])
            pids.append([self._specie_idx[s] for s in self.g.get_products(rxn)])
            ks.append(self.g.get_properties(rxn)[self.rate_constant_property_name])
        # returning the total function
        return lambda t, concs: np.sum([self._reaction_rate_function(t, concs, k, ridxs, pidxs) for k, ridxs, pidxs in zip(ks, rids, pids)], axis=0)
    
    def get_jacobian(self, sp: str):
        """Method to estimate the diagonal element of the Jacobian of the concentration function. namely, we calculate d(d[sp]/dt)/d[sp]"""
        # we care only on the consuming reactions - as these are the only reactions that depened on the concentration of sp
        ks = []
        sp_idxs = []
        for rxn in self.g.consuming_reactions(sp):
            ks.append(self.g.get_properties(rxn)[self.rate_constant_property_name])
            sp_idxs.append([])
            for s in self.g.get_reactants(rxn):
                if s != sp:
                    sp_idxs[-1].append(self._specie_idx[s])
            if len(sp_idxs[-1]) == 0:
                sp_idxs[-1].append(self._specie_idx[sp])
        return lambda concs: np.sum([k * np.prod(concs[idxs]) for k, idxs in zip(ks, sp_idxs)])

    def get_timescale(self, origin: str):
        """Getting the timescale of the system given a single origin node (starting conc=1). givin the time scale for consumption of this specie"""
        concs = self._build_conc_vector(self.g)
        jacobian = self.get_jacobian(origin, self.g)
        avg = 0
        for _ in range(20):
            concs = np.random.rand(len(concs))
            concs[self._specie_idx[origin]] = 1
            concs[0] = 1
            avg += 1 / jacobian(concs)
        return avg / 20

    def solve_kinetics(self, simulation_time: float, timestep: float, initial_concs: List[float], **solver_kwargs):
        """Solve the rate equations at given conditions.
        ARGS:
            - simulation_time (float): total simulation time
            - timestep (float): time of each simulation step
            - initial_concs (List[float]): list of initial specie concentrations
            - **solver_kwargs: keywords for scipy.integrate.ode.set_integrator method
        RETURNS:
            None"""
        # adding null specie to initial concs
        _initial_concs = np.array(list(initial_concs))
        # building target function
        target_f = self._build_target_f()
        # add default solver if non else is specified
        if not "name" in solver_kwargs:
            solver_kwargs = {"name": "lsoda"}
        # setting up solver
        solver = ode(target_f)
        solver.set_integrator(**solver_kwargs)
        solver.set_initial_value(y=_initial_concs)
        # solving the ODE
        t = 0
        self._concs = [_initial_concs]
        self._ts = [0]
        while solver.successful() and t <= simulation_time:
            t = t + timestep
            self._ts.append(t)
            self._concs.append(solver.integrate(t))
            if not solver.successful():
                print("SOLVER WAS NOT SUCCESSFUL")
                return False
        return True

    def concentrations_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._concs, index=self._ts, columns=self.g.get_species())

    def get_concentration(self, sp: str, step: int) -> float:
        return self._concs[step][self._specie_idx[sp]]

    def get_rate(self, rxn: str, step: int) -> float:
        """Method to get the rate of a reaction at a given simulation step"""
        return self.g.get_properties(rxn)[self.rate_constant_property_name] * np.product([self.get_concentration(s, step) for s in self.g.get_reactants(rxn)])

    def get_max_rate(self, rxn: str) -> float:
        """Get the maximal reaction rate"""
        reactants = self.g.get_reactants(rxn)
        df = self.concentrations_df()
        rates = df[reactants].product(axis=1) * self.g.get_properties(rxn)[self.rate_constant_property_name]
        return np.max(rates)
    
    def get_rates(self) ->List[float]:
        df = self.concentrations_df()
        ajr = {}
        for rxn in self.g.get_reactions():
            reactants = self.g.get_reactants(rxn)
            rates = df[reactants].product(axis=1) * self.g.get_properties(rxn)[self.rate_constant_property_name]
            ajr[rxn] = rates
        return pd.DataFrame(ajr, index=df.index)
    

class PreferentialAttachmentModel:


    def __init__(self, n_species: int, n_reactions: int, specie_energy_dist: rv_continuous, alpha: float=1) -> None:
        self.n_species = n_species
        self.n_reactions = n_reactions
        self.alpha = alpha
        self.specie_energy_dist = specie_energy_dist
        if comb(n_species + 1, 4) < n_reactions:
            raise ValueError("Cannot create a network with {} species and {} reactions. try increasing number of species or reduce number of reactions")

    def sample_energy(self) -> float:
        while True:
            x = self.specie_energy_dist.rvs()
            if x > 0:
                return x

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

if __name__ == "__main__":
    from scipy.stats import norm
    # make graph
    nreactions = 1000
    nspecies = 30
    model = PreferentialAttachmentModel(n_species=nspecies, 
                                        n_reactions=nreactions, 
                                        specie_energy_dist=norm(0.4, 0.3))
    og = model.generate_graph()

    # percolate
    n_remove = 100
    counter = 0
    red = og.copy()
    while counter < n_remove:
        rxns = red.get_reactions()
        if len(rxns) == 0:
            break
        node = np.random.choice(red.get_reactions(), 1)[0]
        red.remove_reaction(node)
        counter += 1

    print(pd.DataFrame([[len(x.get_species()), len(x.get_reactions())] for x in [og, red]], columns=["nspecies", "nreactions"], index=["og", "red"]))


