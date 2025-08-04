import torinanet as tn
from typing import List, Union
from sqlalchemy import Column, Integer, String, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class Specie(Base):

    __tablename__ = 'species'

    id = Column(String, primary_key=True)
    gid = Column(String)
    sid = Column(String)
    identifier = Column(String)
    properties = Column(JSON, default={})


class Reaction(Base):

    __tablename__ = 'reactions'

    id = Column(String, primary_key=True)
    gid = Column(String)
    rid = Column(String)
    pretty_string = Column(String)
    properties = Column(JSON, default={})
    r1 = Column(String)
    r2 = Column(String)
    p1 = Column(String)
    p2 = Column(String)

class Graph (Base):

    __tablename__ = "graphs"

    gid = Column(String, primary_key=True)
    kinetic_solver_args = Column(JSON)
    n_species = Column(Integer)
    n_reactions = Column(Integer)


def species_to_sql(rxn_graph: tn.core.RxnGraph, gid: str) -> List[Specie]:
    entries = []
    # Convert species from the reaction graph to Specie SQL entries
    for specie in rxn_graph.species:
        sid = rxn_graph.specie_collection.get_key(specie)
        sql_specie = Specie(
            id=f"{gid}/{sid}",
            gid=gid,
            sid=sid,
            identifier=specie.identifier,
            properties=specie.properties
        )
        entries.append(sql_specie)
    return entries

def reactions_to_sql(rxn_graph: tn.core.RxnGraph, gid: str) -> List[Reaction]:
    entries = []
    # Convert reactions from the reaction graph to Reaction SQL entries
    for reaction in rxn_graph.reactions:
        rid = rxn_graph.reaction_collection.get_key(reaction)
        sql_reaction = Reaction(
            id=f"{gid}/{rid}",
            gid=gid,
            rid=rid,
            pretty_string=reaction.pretty_string(),
            properties=reaction.properties,
        )
        for i, r in enumerate(reaction.reactants):
            setattr(sql_reaction, "r" + str(i + 1), f"{gid}/{rxn_graph.specie_collection.get_key(r)}")
        for i, r in enumerate(reaction.products):
            setattr(sql_reaction, "p" + str(i + 1), f"{gid}/{rxn_graph.specie_collection.get_key(r)}")
        entries.append(sql_reaction)
    return entries


def graph_to_sql(rxn_graph: tn.core.RxnGraph, gid: str) -> Graph:
    return Graph(
        gid=gid, 
        n_species=rxn_graph.get_n_species(),
        n_reactions=rxn_graph.get_n_reactions()
    )

def make_database(path: str) -> Session:
    engine = create_engine("sqlite:///" + path)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()
