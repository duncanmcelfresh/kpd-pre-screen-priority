# KPDData is a container class to keep track of UNOS/KPD-specific data during output

import itertools
from collections import defaultdict

import numpy as np

ABO_LIST = ["a", "b", "ab", "o"]
PAIR_ABO_LIST = list(itertools.product(ABO_LIST, repeat=2))

CYCLE_CAP = 3
CHAIN_CAP = 4
CYCLE_LENGTHS = [x for x in range(2, CYCLE_CAP + 1)]
CHAIN_LENGTHS = [x for x in range(1, CHAIN_CAP + 1)]


class KPDData(object):
    """a class for saving UNOS and KPD_specific data of a kidney exchange graph or matching."""

    # lower edge of the bins are closed, upper edge is open:
    # [in_deg_bin_edges[0], in_deg_bin_edges[1]), [in_deg_bin_edges[1], in_deg_bin_edges[2]), ...
    in_deg_bin_edges = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 1e5]
    out_deg_bin_edges = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 1e5]

    def __init__(self):
        self.donor_abo = {
            abo: 0.0 for abo in ABO_LIST
        }  # number of matched donors of each abo
        self.recip_abo = {
            abo: 0.0 for abo in ABO_LIST
        }  # number of matched recips of each abo
        self.pair_abo = {
            abo: 0.0 for abo in PAIR_ABO_LIST
        }  # number of matched pairs of each abo
        self.ndd_abo = {abo: 0.0 for abo in ABO_LIST}
        self.in_deg_counts = np.zeros(
            len(KPDData.in_deg_bin_edges) - 1
        )  # number of pair vertices with in-degree in a specific bin
        self.out_deg_counts = np.zeros(
            len(KPDData.in_deg_bin_edges) - 1
        )  # number of pair vertices with out-degree in a specific bin
        self.ndd_out_deg_counts = np.zeros(len(KPDData.in_deg_bin_edges) - 1)
        self.high_low_sensitized_count = np.zeros(
            2
        )  # high_low_sensitized_count[0] and [1] = number of highly- and
        # lowly-sensitized patients

        self.cycle_counts = {x: 0.0 for x in CYCLE_LENGTHS}
        self.chain_counts = {x: 0.0 for x in CHAIN_LENGTHS}

    def __add__(self, other):
        new_data = KPDData()
        new_data.donor_abo = {
            abo: self.donor_abo[abo] + other.donor_abo[abo] for abo in ABO_LIST
        }
        new_data.recip_abo = {
            abo: self.recip_abo[abo] + other.recip_abo[abo] for abo in ABO_LIST
        }
        new_data.ndd_abo = {
            abo: self.ndd_abo[abo] + other.ndd_abo[abo] for abo in ABO_LIST
        }
        new_data.pair_abo = {
            abo: self.pair_abo[abo] + other.pair_abo[abo] for abo in PAIR_ABO_LIST
        }
        # add up cycle and chain counts
        for k, v in self.cycle_counts.items():
            other_v = other.cycle_counts[k]
            new_data.cycle_counts[k] = v + other_v
        for k, v in self.chain_counts.items():
            other_v = other.chain_counts[k]
            new_data.chain_counts[k] = v + other_v

        new_data.in_deg_counts = np.array(self.in_deg_counts) + np.array(
            other.in_deg_counts
        )
        new_data.out_deg_counts = np.array(self.out_deg_counts) + np.array(
            other.out_deg_counts
        )
        new_data.ndd_out_deg_counts = np.array(self.ndd_out_deg_counts) + np.array(
            other.ndd_out_deg_counts
        )

        new_data.high_low_sensitized_count = np.array(
            self.high_low_sensitized_count
        ) + np.array(other.high_low_sensitized_count)

        return new_data

    def __truediv__(self, x):
        new_data = KPDData()
        new_data.donor_abo = {abo: self.donor_abo[abo] / x for abo in ABO_LIST}
        new_data.recip_abo = {abo: self.recip_abo[abo] / x for abo in ABO_LIST}
        new_data.ndd_abo = {abo: self.ndd_abo[abo] / x for abo in ABO_LIST}
        new_data.pair_abo = {abo: self.pair_abo[abo] / x for abo in PAIR_ABO_LIST}
        new_data.in_deg_counts = np.array(self.in_deg_counts) / x
        new_data.out_deg_counts = np.array(self.out_deg_counts) / x
        new_data.ndd_out_deg_counts = np.array(self.ndd_out_deg_counts) / x

        for k, v in self.cycle_counts.items():
            new_data.cycle_counts[k] = v / x
        for k, v in self.chain_counts.items():
            new_data.chain_counts[k] = v / x

        new_data.high_low_sensitized_count = (
            np.array(self.high_low_sensitized_count) / x
        )

        return new_data

    def __mul__(self, x):
        new_data = KPDData()
        new_data.donor_abo = {abo: self.donor_abo[abo] * x for abo in ABO_LIST}
        new_data.recip_abo = {abo: self.recip_abo[abo] * x for abo in ABO_LIST}
        new_data.ndd_abo = {abo: self.ndd_abo[abo] * x for abo in ABO_LIST}
        new_data.pair_abo = {abo: self.pair_abo[abo] * x for abo in PAIR_ABO_LIST}
        new_data.in_deg_counts = np.array(self.in_deg_counts) * x
        new_data.out_deg_counts = np.array(self.out_deg_counts) * x
        new_data.ndd_out_deg_counts = np.array(self.ndd_out_deg_counts) * x

        for k, v in self.cycle_counts.items():
            new_data.cycle_counts[k] = v * x
        for k, v in self.chain_counts.items():
            new_data.chain_counts[k] = v * x

        new_data.high_low_sensitized_count = (
            np.array(self.high_low_sensitized_count) * x
        )

        return new_data

    def to_string(self, delimiter=";", end="\n"):
        """return a string representation of the result for CSV-formatted output. this uses the same col ordering
        as header_string()"""
        donor_abo_list = [str(self.donor_abo[abo]) for abo in ABO_LIST]
        recip_abo_list = [str(self.recip_abo[abo]) for abo in ABO_LIST]
        ndd_abo_list = [str(self.ndd_abo[abo]) for abo in ABO_LIST]
        pair_abo_list = [str(self.pair_abo[abo]) for abo in PAIR_ABO_LIST]

        in_deg_list = [str(x) for x in self.in_deg_counts]
        out_deg_list = [str(x) for x in self.out_deg_counts]
        ndd_out_deg_list = [str(x) for x in self.ndd_out_deg_counts]
        cycles_list = [str(self.cycle_counts[i]) for i in CYCLE_LENGTHS]
        chains_list = [str(self.chain_counts[i]) for i in CHAIN_LENGTHS]
        high_low_sensitized_count = [str(x) for x in self.high_low_sensitized_count]
        return (
            delimiter.join(
                donor_abo_list
                + recip_abo_list
                + ndd_abo_list
                + pair_abo_list
                + in_deg_list
                + out_deg_list
                + ndd_out_deg_list
                + cycles_list
                + chains_list
                + high_low_sensitized_count
            )
            + end
        )

    def col_list(col_prefix="matched"):
        """return a list of column strings for the CSV-formatted output"""
        donor_abo_list = [f"{col_prefix}_donor_abo_{abo}" for abo in ABO_LIST]
        recip_abo_list = [f"{col_prefix}_recip_abo_{abo}" for abo in ABO_LIST]
        ndd_abo_list = [f"{col_prefix}_ndd_abo_{abo}" for abo in ABO_LIST]
        pair_abo_list = [
            f"{col_prefix}_pair_abo_{abo[0]}_{abo[1]}" for abo in PAIR_ABO_LIST
        ]
        in_deg_list = [
            f"{col_prefix}_pair_deg_{KPDData.in_deg_bin_edges[i]}_{KPDData.in_deg_bin_edges[i + 1]}"
            for i in range(len(KPDData.in_deg_bin_edges) - 1)
        ]
        out_deg_list = [
            f"{col_prefix}_pair_out_deg_{KPDData.out_deg_bin_edges[i]}_{KPDData.out_deg_bin_edges[i + 1]}"
            for i in range(len(KPDData.out_deg_bin_edges) - 1)
        ]
        ndd_out_deg_list = [
            f"{col_prefix}_ndd_out_deg_{KPDData.out_deg_bin_edges[i]}_{KPDData.out_deg_bin_edges[i + 1]}"
            for i in range(len(KPDData.out_deg_bin_edges) - 1)
        ]
        cycles_list = [f"{col_prefix}_{i}_cycles" for i in CYCLE_LENGTHS]
        chains_list = [f"{col_prefix}_{i}_chains" for i in CHAIN_LENGTHS]
        high_low_sensitized_count = [
            f"{col_prefix}_highly_sensitized_patients",
            f"{col_prefix}_lowly_sensitized_patients",
        ]
        return (
            donor_abo_list
            + recip_abo_list
            + ndd_abo_list
            + pair_abo_list
            + in_deg_list
            + out_deg_list
            + ndd_out_deg_list
            + cycles_list
            + chains_list
            + high_low_sensitized_count
        )
