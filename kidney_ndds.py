# non-directed donor objects and edges
from kidney_digraph import stable_hash

class Ndd:
    """A non-directed donor"""

    def __init__(self, id=None, aux_id=None):
        self.edges = []
        self.chain_weight = 0
        self.id = id
        self.aux_id = aux_id
        self.out_degree = 0
        self.data = {}

    def add_edge(self, ndd_edge):
        """Add an edge representing compatibility with a patient who appears as a
        vertex in the directed graph."""
        self.out_degree += 1
        self.edges.append(ndd_edge)

    def get_edge(self, tgt_idx):
        """
        return the edge that points to vertex with id tgt_idx, if it exists
        """
        tgt_edges = [e for e in self.edges if e.tgt.id == tgt_idx]
        if len(tgt_edges) == 1:
            return tgt_edges[0]
        elif len(tgt_edges) == 0:
            raise Warning("edge not found to target index %d" % tgt_idx)
        else:
            raise Warning("multiple edges found to target index %d" % tgt_idx)


class NddEdge:
    """An edge pointing from an NDD to a vertex in the directed graph"""

    def __init__(
        self,
        tgt,
        weight,
        src_id=None,
        src=None,
        data={},
    ):
        self.tgt = tgt
        self.src = src
        tgt.in_degree += 1
        self.tgt_id = tgt.id
        self.src_id = src_id
        self.weight = weight  # edge weight
        self.success = True
        self.sensitized = tgt.sensitized
        self.data = data

        # - edge-query-specific properties -
        self.p_reject = 0.0  # probability of rejection
        self.p_success_accept = 1.0  # probability of success, if accepted
        self.p_success_noquery = 1.0  # probability of success, if not queried

        self.hash_str, self.hash_int = stable_hash(
            (self.src_id, self.tgt_id, "NddEdge")
        )

    def __str__(self):
        return "NDD edge to V{}".format(self.tgt.id)

    def __eq__(self, other):
        return self.hash_int == other.hash_int

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return self.hash_int


class Chain(object):
    """A chain initiated by an NDD.
    
    Data members:
        ndd_index: The index of the NDD
        vtx_indices: The indices of the vertices in the chain, in order
        weight: the chain's weight
        edges: ordered list of the chain's edges
    """

    def __init__(self, ndd_index, vtx_indices, weight):
        self.ndd_index = ndd_index
        self.vtx_indices = vtx_indices
        self.weight = weight

    @property
    def length(self):
        return len(self.vtx_indices)

    def __repr__(self):
        return (
            "Chain NDD{} ".format(self.ndd_index)
            + " ".join(str(v) for v in self.vtx_indices)
            + " with weight "
            + str(self.weight)
        )

    def __cmp__(self, other):
        # Compare on NDD ID, then chain length, then on weight, then
        # lexicographically on vtx indices
        if self.ndd_index < other.ndd_index:
            return -1
        elif self.ndd_index > other.ndd_index:
            return 1
        elif len(self.vtx_indices) < len(other.vtx_indices):
            return -1
        elif len(self.vtx_indices) > len(other.vtx_indices):
            return 1
        elif self.weight < other.weight:
            return -1
        elif self.weight > other.weight:
            return 1
        else:
            for i, j in zip(self.vtx_indices, other.vtx_indices):
                if i < j:
                    return -1
                elif i > j:
                    return 1
        return 0

    def get_edge_objs(self, digraph, ndds):
        """
        to get edges from a chain:
        first get the ndd edge into the digraph using ndd.get_edge(chain.vtx_indices[0])
        then get the edges in the graph object using graph.adj_mat
        """
        ndd = ndds[self.ndd_index]
        tgt_id = self.vtx_indices[0]

        edges = [ndd.get_edge(tgt_id)]

        for i_src in range(len(self.vtx_indices) - 1):
            curr_vtx = self.vtx_indices[i_src]
            next_vtx = self.vtx_indices[i_src + 1]
            edges.append(digraph.adj_mat[curr_vtx][next_vtx])
        return edges
