# class for representing exchange grpah
import kidney_utils

from kidney_digraph import Cycle
from kidney_ip import OptConfig, create_picef_model


class GraphStructure(object):
    """contains an instance of the kidney exchange problem. also includes matching parameters (cycle/chain cap) and
    edge failure probabilities used when constructing the matching."""

    def __init__(
        self,
        graph,
        altruists,
        cycle_cap,
        chain_cap,
        name=None,
        df_donor=None,
        df_recip=None,
        logger=None,
    ):

        self.graph = graph
        self.altruists = altruists
        self.cycle_cap = cycle_cap
        self.chain_cap = chain_cap
        self.all_edge_list = list(self.all_edges())
        self.name = name
        self.logger = logger
        self.df_donor = df_donor
        self.df_recip = df_recip
        self.init_edge_ids()
        self.init_vertex_ids()
        self.init_matchable()
        self.matchable_edge_list = [e for e in self.all_edge_list if e.matchable]

        self.init_optconfig()

    def init_optconfig(self, edge_success_prob=1.0):
        # initialize OptConfig for this graph
        self.optconfig = OptConfig(
            self.graph,
            self.altruists,
            self.cycle_cap,
            self.cycle_cap,
            edge_success_prob=edge_success_prob,
        )
        # add the gurobi model to the OptConfig object
        create_picef_model(self.optconfig)

    def init_matchable(self):
        """initialize the property matchable for each edge in the graph. matchable = True means the edge can be included
        in any matching (i.e., it's in at least one cycle or chain)"""

        # initialize all edges to be not-matchable
        for e in self.all_edge_list:
            e.matchable = False

        cycles = self.graph.find_cycles(self.cycle_cap)
        cycle_list = []
        # all edges in a cycle are matchable
        for c in cycles:
            c_obj = Cycle(c)
            c_obj.add_edges(self.graph.es)
            cycle_list.append(c_obj)
            for e in c_obj.edges:
                e.matchable = True

        # all NDDEdges are matchable
        for n in self.altruists:
            for e in n.edges:
                e.matchable = True

        # all pair-pair edges that are within chain distance are matchable
        dists_from_ndd = kidney_utils.get_dist_from_nearest_ndd(
            self.graph, self.altruists
        )
        for e in self.all_edge_list:
            if dists_from_ndd[e.src.id] <= self.chain_cap - 1:
                e.matchable = True

        if self.logger is not None:
            self.logger.info(
                f"matchable edges: {len([e for e in self.all_edge_list if e.matchable])} (out of {len(self.all_edge_list)} total)"
            )

    def init_edge_ids(self):
        """initialize edge index property to be unique for each edge in the graph"""
        for i, edge in enumerate(self.all_edge_list):
            edge.index = i

    def init_vertex_ids(self):
        """initialize vertex index property to be unique for each edge in the graph"""
        for i, ndd in enumerate(self.altruists):
            ndd.index = i
        for i, v in enumerate(self.graph.vs):
            v.index = i + len(self.altruists)

    def all_edges(self):
        """yield edges in order, first digraph then ndd"""
        for edge in self.graph.es:
            yield edge
        for ndd in self.altruists:
            for edge in ndd.edges:
                yield edge
