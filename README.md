# Token Elimination Reproducibility Package

This is a reproducibility package for the paper 'Token Elimination in Model Checking of Petri Nets' TACAS'25 by Nicolaj Ø. Jensen, Kim G. Larsen, and Jiri Srba.

The package contains the models, queries, and binary used to produce the results of the paper as well as scripts to reproduce the graphs and tables.
Due to the size of the benchmark, reproducing the data takes a significant amount of time, and we therefore also include the data used for the paper's graphs and tables in the package.

The package was tested using WSL2 and [TACAS'23 AE VM](https://zenodo.org/records/7113223).

DOI: 10.5281/zenodo.14608439

## Prerequisites

- Bash
- Python 3.10 (`python3` and `pip`)
- tar

## Instructions

### Setup

*Expected run time: 4 minutes (shorter on subsequent runs)*

Steps:

- Run `chmod +x scripts/*`
- Optionally set up a Python virtual environment:
  - `python3 -m venv venv`
  - `source venv/bin/activate`
- Run `./scripts/setup.sh` (*Expected run time: 4 minutes*)

### Test pipeline (Short early review)

*Expected run time: 2 minutes*

Run `./scripts/run_mini.sh`

This pipeline is intended for verifying that the binary and bash files work.
The pipeline only runs a few queries in total and will not produce meaningful results.

### Reproduce data (full)

*Expected run time: 200-300 days*

Run `./scripts/run_full.sh`

This pipeline runs each query in `MCC2023-CTL` and will take a _very long time_.
Therefore, we have included the data files used in the paper in this reproducibility package: `data/demo_tapaal.csv`, `data/demo_dynamic.csv`, and `data/demo_static.csv`.
You may also consider producing the data partially. See the section below.

If you wish to rerun the entire benchmark, we recommend running the queries in parallel. However, we do not include scripts to do so.
See `scripts/run_single.sh` to run a single query and `scripts/extract.sh` to extract the data from the log files.
The time/memory limits can be found in `scripts/run_single.sh` (30 minutes and 15 GB by default).

### Reproduce data (partial)

*Expected run time: 6 hours*

Run `./scripts/run_partial.sh`

This partial pipeline runs the first query of every 15th model with a timeout of 10 minutes.
Note that the shorter timeout may be a disadvantage for the dynamic and static token-elimination methods.

### Generate graphs and tables

*Expected run time: 20 seconds*

Steps:

- Run `python3 scripts/graphs_and_tables.py`
  - By default, the demo data (the data used in the paper) is used. To use your own data, the Python script must be
    given a series of arguments on the form 'name=file' where 'name' is the display name of the data and 'file' is
    the name of the file in `data/`. You must provide at least two data files and the first one must be named Tapaal
    as it will be used as the baseline for some graphs. If you did not modify the pipeline scripts, use the command:
    `python3 graphs_and_tables.py Tapaal=ae_tapaal.csv Static=ae_static.csv Dynamic=ae_dynamic.csv`.
  - Some deprecation warnings may appear. Those are expected.
- Graphs and tables can now be found in `output/`.

## Licensing Information

### Verifypn

This artifact includes a compiled binary of the tool verifypn, `bin/verifypn-tokelim-linux64`, which implements the techniques described in the accompanying paper.
The binary is distributed under the terms of the GNU General Public License v3.0.

You can find the source code for verifypn, including the version used to produce this binary, at the following repository: https://github.com/NicEastvillage/verifypn/tree/token_elim_good.

For more details on the GPL v3 license, please refer to the included `bin/LICENSE` file, or visit https://www.gnu.org/licenses/gpl-3.0.html.

### MCC models & queries

The models and queries found in `MCC2023-CTL.tar.gz` is a subset of the models and queries from the Model Checking Contest 2023.
Spefically, the tarball contains the `pnml` Petri net model files and the CTL cardinality and fireability queries in `xml` form.
Further details about the Model Checking Contest 2023 as well as the models and queries can be found at https://mcc.lip6.fr/2023/.

### Other

The remaining parts of this reproducibility package such as the scripts are distributed under the MIT license. See `LICENSE`.
