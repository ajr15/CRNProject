from scipy.integrate import ode, solve_ivp
import numpy as np
import pandas as pd
from typing import List, Optional
from src.SimulatedReactionGraph import SimulatedReactionGraph

class KineticSolver:

    """Kinetic solver for a generated reaction graph"""

    def __init__(self, g: SimulatedReactionGraph, rate_constant_property_name: str="k"):
        self.g = g
        self.rate_constant_property_name = rate_constant_property_name
        self._specie_idx = {sp: i + 1 for i, sp in enumerate(g.get_species())}
        self._concs = []
        self._ts = []


    # @staticmethod
    # def _reaction_rate_function(t, concs, k, ridxs, pidxs):
    #     diff = np.zeros(len(concs))
    #     r = k * np.prod([concs[i] for i in ridxs])
    #     for i in ridxs:
    #         diff[i] -= r
    #     for i in pidxs:
    #         diff[i] += r
    #     return diff

    # def _build_target_f(self):
    #     """build the function for the solver"""
    #     ks = []
    #     rids = []
    #     pids = []
    #     for rxn in self.g.get_reactions():
    #         rids.append([self._specie_idx[s] for s in self.g.get_reactants(rxn)])
    #         pids.append([self._specie_idx[s] for s in self.g.get_products(rxn)])
    #         ks.append(self.g.get_properties(rxn)[self.rate_constant_property_name])
    #     # returning the total function
    #     return lambda t, concs: np.sum([self._reaction_rate_function(t, concs, k, ridxs, pidxs) for k, ridxs, pidxs in zip(ks, rids, pids)], axis=0)
    
    # def _build_vectorized_target_f(self):
    #     """build the function for the solver"""
    #     rate_mat = self._rate_matrix()
    #     return lambda t, concs: rate_mat @ (concs @ concs.T)
    
    # def get_jacobian(self, sp: str):
    #     """Method to estimate the diagonal element of the Jacobian of the concentration function. namely, we calculate d(d[sp]/dt)/d[sp]"""
    #     # we care only on the consuming reactions - as these are the only reactions that depened on the concentration of sp
    #     ks = []
    #     sp_idxs = []
    #     for rxn in self.g.consuming_reactions(sp):
    #         ks.append(self.g.get_properties(rxn)[self.rate_constant_property_name])
    #         sp_idxs.append([])
    #         for s in self.g.get_reactants(rxn):
    #             if s != sp:
    #                 sp_idxs[-1].append(self._specie_idx[s])
    #         if len(sp_idxs[-1]) == 0:
    #             sp_idxs[-1].append(self._specie_idx[sp])
    #     return lambda concs: np.sum([k * np.prod(concs[idxs]) for k, idxs in zip(ks, sp_idxs)])

    # def get_timescale(self, origin: str):
    #     """Getting the timescale of the system given a single origin node (starting conc=1). givin the time scale for consumption of this specie"""
    #     concs = self._build_conc_vector(self.g)
    #     jacobian = self.get_jacobian(origin, self.g)
    #     avg = 0
    #     for _ in range(20):
    #         concs = np.random.rand(len(concs))
    #         concs[self._specie_idx[origin]] = 1
    #         concs[0] = 1
    #         avg += 1 / jacobian(concs)
    #     return avg / 20

    # def solve_kinetics(self, simulation_time: float, timestep: float, initial_concs: List[float], **solver_kwargs):
    #     """Solve the rate equations at given conditions.
    #     ARGS:
    #         - simulation_time (float): total simulation time
    #         - timestep (float): time of each simulation step
    #         - initial_concs (List[float]): list of initial specie concentrations
    #         - **solver_kwargs: keywords for scipy.integrate.ode.set_integrator method
    #     RETURNS:
    #         None"""
    #     # adding null specie to initial concs
    #     _initial_concs = np.array(list(initial_concs))
    #     # building target function
    #     target_f = self._build_target_f()
    #     # add default solver if non else is specified
    #     if not "name" in solver_kwargs:
    #         solver_kwargs = {"name": "lsoda"}
    #     # setting up solver
    #     solver = ode(target_f)
    #     solver.set_integrator(**solver_kwargs)
    #     solver.set_initial_value(y=_initial_concs)
    #     # solving the ODE
    #     t = 0
    #     self._concs = [_initial_concs]
    #     self._ts = [0]
    #     while solver.successful() and t <= simulation_time:
    #         t = t + timestep
    #         self._ts.append(t)
    #         self._concs.append(solver.integrate(t))
    #         if not solver.successful():
    #             print("SOLVER WAS NOT SUCCESSFUL")
    #             return False
    #     return True
    
    # def solve_kinetics_ivp(self, initial_concs: List[float], ss_threshold: float=1e-10, max_t: Optional[float]=None, atol: float=1e-10, rtol: float=1e-3, method: str="BDF"):
    #     """Solve the rate equations at given conditions.
    #     ARGS:
    #         - simulation_time (float): total simulation time
    #         - timestep (float): time of each simulation step
    #         - initial_concs (List[float]): list of initial specie concentrations
    #         - **solver_kwargs: keywords for scipy.integrate.ode.set_integrator method
    #     RETURNS:
    #         None"""
    #     # building target function
    #     target_f = self._build_target_f()
    #     ss_event = lambda t, y: 0 if np.abs(np.max(target_f(t, y))) < ss_threshold else 1
    #     ss_event.terminal = True
    #     if max_t is None:
    #         max_t = np.max([1 / rxn.properties[self.rate_constant_property_name] for rxn in self.g.reactions()])
    #     sol = solve_ivp(fun=target_f, t_span=(0, max_t), y0=initial_concs, method=method, events=ss_event, atol=atol, rtol=rtol)
    #     self._ts = sol.t
    #     self._concs = sol.y.T

    def _rate_matrix(self):
        """Trying to make a more efficient target f"""
        nspecies = len(self._specie_idx) + 1
        nreactions = len(self.g.get_reactions())
        # building the selector matrix
        Q = np.zeros((nreactions, nspecies ** 2))
        A = np.zeros((nspecies, nreactions))
        for i, rxn in enumerate(self.g.get_reactions()):
            rate_constant = self.g.get_properties(rxn)[self.rate_constant_property_name]
            # find the required selection index - the reactants product
            rids = [self._specie_idx[s] for s in self.g.get_reactants(rxn)]
            if len(rids) == 1:
                rids.append(0)
            m, k = rids
            Q[i, m * nspecies + k] = 1 # select the m, k index from the concentrations vector
            # now build the rate matrix
            pids = [self._specie_idx[s] for s in self.g.get_products(rxn)]
            for p in pids:
                A[p, i] = rate_constant
            for r in rids:
                if r != 0:
                    A[r, i] = - rate_constant
        return A @ Q
    

    @staticmethod
    def jacobian(t, concs, M):
        # ajr = np.concatenate([[1], concs.flatten("F")]).reshape(-1, 1)
        ajr = concs.flatten("F").reshape(-1, 1)
        n = len(ajr)
        I = np.eye(n)
        kron1 = np.kron(I, ajr)
        kron2 = np.kron(ajr, I)
        return M @ (kron1 + kron2)
    
    @staticmethod
    def _target_function(t, concs, M):
        # ajr = np.concatenate([[1], concs.flatten("F")]).reshape(-1, 1)
        ajr = concs.flatten("F").reshape(-1, 1)
        return (M @ np.kron(ajr, ajr).reshape(-1, 1)).flatten("F")


    def solve_kinetics(self, initial_concs: List[float], ss_threshold: float=1e-10, max_t: Optional[float]=None, atol: float=1e-10, rtol: float=1e-3, method: str="BDF"):
        """Solve the rate equations at given conditions.
        ARGS:
            - simulation_time (float): total simulation time
            - timestep (float): time of each simulation step
            - initial_concs (List[float]): list of initial specie concentrations
            - **solver_kwargs: keywords for scipy.integrate.ode.set_integrator method
        RETURNS:
            None"""
        # building target function
        rate_mat = self._rate_matrix()
        target_f = lambda t, concs: self._target_function(t, concs, rate_mat)
        jac = lambda t, concs: self.jacobian(t, concs, rate_mat)
        ss_event = lambda t, y: 0 if np.abs(np.max(target_f(t, y))) < ss_threshold else 1
        ss_event.terminal = True
        if max_t is None:
            max_t = np.max([1 / rxn.properties[self.rate_constant_property_name] for rxn in self.g.reactions()])
        iconcs = np.array([1] + initial_concs)
        sol = solve_ivp(fun=target_f, jac=jac, t_span=(0, max_t), y0=iconcs, method=method, events=ss_event, atol=atol, rtol=rtol)
        self._ts = sol.t
        self._concs = sol.y.T[:, 1:]

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
    
    def rates_df(self) -> pd.DataFrame:
        df = self.concentrations_df()
        ajr = {}
        for rxn in self.g.get_reactions():
            reactants = self.g.get_reactants(rxn)
            rates = df[reactants].product(axis=1) * self.g.get_properties(rxn)[self.rate_constant_property_name]
            ajr[rxn] = rates
        return pd.DataFrame(ajr, index=df.index)
    
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    g = SimulatedReactionGraph.from_file("data/simulated/s25r1000/26.json")
    solver = KineticSolver(g)
    iconcs = [1 if s in g.origin else 0 for s in g.get_species()]
    solver.solve_kinetics_imporoved_ivp(initial_concs=iconcs)
    df = solver.concentrations_df()
    print(df)
    for c in df.columns:
        if df[c].mean() > 0.001:
            plt.plot(df.index, df[c], label=c)
    plt.xscale("log")
    plt.savefig("test.png")