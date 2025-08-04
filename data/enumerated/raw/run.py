import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import os
import argparse
from sqlalchemy import Column, Integer, String

import enumerator
import torinanet as tn
import torinax as tx
from torinax.pipelines.computations import comp_sql_model_creator


COMP_ARGS = {
    "ammonia": {
        "reactants": [tn.core.Specie("O=O"), tn.core.Specie("N")],
        "n_iterations": 4,
        "ac_filters": [
            tn.iterate.ac_matrix_filters.MaxAtomsOfElement({7: 2, 8: 2}), 
            tn.iterate.ac_matrix_filters.MaxBondsPerAtom(), 
            tn.iterate.ac_matrix_filters.MaxComponents(2),
            tn.iterate.ac_matrix_filters.MaxRingNumber(0),
        ]
    },
    "hydrogen": {
        "reactants": [tn.core.Specie("O=O"), tn.core.Specie("[H][H]")],
        "n_iterations": 8,
        "ac_filters": [
            tn.iterate.ac_matrix_filters.MaxAtomsOfElement({7: 2, 8: 3}), 
            tn.iterate.ac_matrix_filters.MaxBondsPerAtom(), 
            tn.iterate.ac_matrix_filters.MaxComponents(2),
            tn.iterate.ac_matrix_filters.MaxRingNumber(1),
        ]
    }
}


if __name__ == "__main__":
    # argument parser for different netnworks
    parser = argparse.ArgumentParser()
    parser.add_argument("reaction", type=str, help="reaction to enumerate. options: {}".format(", ".join(COMP_ARGS.keys())))
    parser.add_argument("--rundir", type=str, default=None, help="directory to make the run in")
    args = parser.parse_args()
    # init
    if args.rundir is None:
        args.rundir = "./{}".format(args.reaction)
    if not os.path.isdir(args.rundir):
        os.mkdir()
    os.chdir(args.rundir)
    rxn_graph = tn.core.RxnGraph()
    # setting source species in graph
    rxn_graph.set_source_species(COMP_ARGS[args.reaction]["reactants"], force=True)

    # set default conversion filters
    max_breaking_bonds = 2
    max_forming_bonds = 2
    conversion_filters = [tn.iterate.conversion_matrix_filters.MaxChangingBonds(max_breaking_bonds + max_forming_bonds),
                                    tn.iterate.conversion_matrix_filters.MaxFormingAndBreakingBonds(max_forming_bonds, max_breaking_bonds),
                                    tn.iterate.conversion_matrix_filters.OnlySingleBonds()]

    # set default ac filters
    ac_filters = COMP_ARGS[args.reaction]["ac_filters"]

    # make log table model (for using the current implementation)
    x = comp_sql_model_creator("log", {"id": Column(String, primary_key=True), "iteration": Column(Integer), "source": Column(String)})

    # set run
    pipeline = [enumerator.computations.ElementaryReactionEnumeration(conversion_filters=conversion_filters, ac_filters=ac_filters)]
    # getting threshold
    print("RUNNING FOR {}".format(args.reaction))
    enumer = enumerator.Enumerator.Enumerator(rxn_graph, pipeline, COMP_ARGS[args.reaction]["n_iterations"], ".", reflect=False)
    enumer.enumerate()
