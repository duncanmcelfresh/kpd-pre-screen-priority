# kpd-pre-screen-priority

Code for calculating the pre-screening priority of each edge in a UNOS KPD match run.

**Dependencies:**
- python 3
- numpy
- pandas
- [Gurobi](https://www.gurobi.com) optimizer and gurobipy

### Overview

This code runs the "Greedy" edge selection algorithm from [Improving Policy-Constrained Kidney Exchange via Pre-Screening](https://arxiv.org/abs/2010.12069) (NeurIPS'20) for a UNOS match run. The only required input is a file describing the available transplants in the match run, represented by a file in the standard UNOS format ("`...edgeweights.csv`"). 

This code (`calculate_edge_prescreen_priority.py`) writes two files: (a) a "pre-screen priority" CSV file with one line per edge, with a "prescreen score" indicating how important it is to pre-screen this edge, and (b) a log file with any warning and error messages. 

The pre-screen priority CSV has three columns:
- `KPD_candidate_id`: this is exactly the KPD candidate ID read from the `...edgeweights.csv` file.
- `KPD_donor_id`: this is exactly the KPD donor ID read from the `...edgeweights.csv` file.
- `prescreen_score`: the "score" of the edge. 

The score of each edge has the following meaning:
- score = -1: the edge cannot be matched in a feasible cycle or chain.
- score = 0: the edge can be matched in a feasible cycle or chain, but it was not selected by our algorithm.
- score > 0: the edge was selected by our algorithm, and higher scores mean higher priority.

#### Important Inputs

There are three input parameters that change the behavior of the edge selection, and they should be selected carefully. The default values of these parameters are reasonable, but additional sensitivity analysis may be warranted:
- `--p-reject`:  an estimate of the probability that edges are pre-rejected if they are screened by the recipient.
- `--p-success-accept`: probability that an edge in the final match run will succeed if accepted during pre-screening.
- `--p-success-noquery`: probability that an edge in the final match run will succeed if it is not pre-screened by the recipient.


## Example Usage

Edge pre-screen priority can be calculated by running the script `calculate_edge_prescreen_priority.py`, and passing the following input parameters. For example, if you want to find the 200 most-important edges to pre-screen, you can run:

```
python -m calculate_edge_prescreen_priority.py --num-prescreen-edges 200 --out-dir /output/dir/ --kpd-dir /unos/data/KPD_CSV_IO_20160602
```

And this will write two files: a log file, with path similar to `/output/dir/LOGS_###.txt`, and an edge priority file with path `/output/dir/prescreen_priority_###.csv`. (The "###" will be a string representing the date/time.) Edges in the resulting CSV file will be sorted from highest-priority to lowest-priority.

The prescreen priority file will look like the following:

```
KPD_candidate_id,KPD_donor_id,candidate_ctr,donor_ctr,prescreen_score
203746,603996,GAMC-TX1,MIUM-TX1,10
203832,603651,MNHC-TX1,WISE-TX1,9
203852,603867,MNAN-TX1,CASM-TX1,8
203860,603863,FLFH-TX1,PALV-TX1,7
203735,604002,TXTX-TX1,MNHC-TX1,6
203832,604019,MNHC-TX1,NCBG-TX1,5
203515,603895,WISE-TX1,TXTX-TX1,4
203851,603910,FLFH-TX1,GAMC-TX1,3
203829,604023,MIUM-TX1,FLFH-TX1,2
203696,603917,CASM-TX1,NYFL-TX1,1
203832,603986,MNHC-TX1,WISE-TX1,0
203746,601221,GAMC-TX1,TXAS-TX1,0
...
203829,601300,MIUM-TX1,PALV-TX1,-1
203829,601305,MIUM-TX1,TXTX-TX1,-1
203829,601329,MIUM-TX1,UTMC-TX1,-1
```

Note that most edges will have score -1, meaning they cannot be matched.

## All Parameters

Input/output parameters
- `--out-dir`: (string, required) relative or absolute path of the directory for writing the edge selection prioritization file and log file.
- `--kpd-dir`: (string, required) relative or absolute path of the directory containing the UNOS match run files, which must have a file named "...edgeweights.csv".

Kidney-exchange parameters
- `--chain-cap`: (int, default 4) maximum chain length for the match run.
- `--cycle-cap`: (int, default 3) maximum cycle length for the match run.

Parameters to adjust the pre-screening priority algorithm
- `--num-prescreen-edges`: (integer, default 200) number of edges to search for.
- `--time-limit`: (int, default 12000) maximum time for the algorithm, in seconds. Recommended to allow about 60 seconds for each edge to search for, however the code usually does not take this long to run. So if ``num-prescreen-edges`` is 100, then `time-limit` should be about 6000.
- `--p-reject`: (float, default 0.1) an estimate of the probability that edges are pre-rejected if they are screened by the recipient.
- `--p-success-accept`: (float, default 0.8) probability that an edge in the final match run will succeed if accepted during pre-screening.
- `--p-success-noquery`: (float, default 0.5) probability that an edge in the final match run if it is not pre-screened by the recipient.

Parameters that probably shouldn't be changed
- `--edge-success-prob`: (float, default 1.0) probability that each edge succeeds, used by the UNOS matching algorithm. This should be 1.0 unless you're testing variants of the UNOS match run algorithm.
- `--num-leaf-samples`: (int, default 200) used to evaluate the value gained by pre-screening an edge. Should be at least 200. Much higher values make the value esimation slightly more accurate.
- `--max-level-for-pruning`: (int, default 4) used to find possible edges to pre-screen. Larger values speed up computation time slightly, but require exponentially more memory; not recommended to use values larger than 8).


## Code Overview

The main script is `calculate_edge_prescreen_priority.py`, which takes as input a directory of UNOS match run files, and writes an edge priority file. Here is a brief outline of the other files

- `edge_selection.py`: code for the edge selection function
- `graphstructure.py`: structure for representing kidney exchange graphs and associated data
- `gurobi_functions.py`: functions for interfacing with gurobipy
- `kidney_digraph.py`: classes for the directed graph representation of kidney exchange
- `kidney_ndds.py`: classes for non-directed donors and their edges
- `kidney_ip.py`: functions for solving the kidney exchange problem using integer programming
- `kidney_utils.py`: helper functions for kidney exchange
- `utils.py`: general helper functions
