# Token Elimination Reproducibility Package

This is a reproducibility package for the paper 'Token Elimination in Model Checking of Petri Nets' by Nicolaj Ø. Jensen, Jiri Srba, and Kim G. Larsen.

The package contains the models, queries, and binary used to produce the results of the paper as well as scripts to reproduce the graphs and tables.
Due to the size of the benchmark, reproducing the data takes a significant amount of time, and we therefore include the data used for the paper's graphs and tables in the package.

The package was tested using WSL2.

## Prerequisites

- Bash
- Python 3.10
- tar

## Instructions

### Setup

Setup permissions:
- For the Bash scripts: `chmod u+x scripts/*.sh`
- For the Python scripts: `chmod u+x scripts/*.py`
- For the verifypn binary: `chmod u+x bin/*`

Setup to reproduce data:
- Extract models and queries: `tar -xvf MCC2023-CTL.tar.gz`

Setup to construct graphs and tables:
- Setup Python virtual environment: `python -m venv .venv`
- Activate the virtual environment: `source .venv/bin/activate`
- Install Python packages: `pip install -r requirements.txt`

### Reproduce data (full)

*Expected run time: 200-300 days per method*

Steps:

- Navigate to the `scripts/` directory: `cd scripts`
- Run pipeline: `./run_pipeline.sh <name> ../bin/verifypn-tokelim-linux64 <method>` where `<name>` is the desired name of the output, and `<method>` is either `tapaal`, `dynamic`, or `static`. We advice to include the method in the name too, e.g. 'ae_tapaal'.
  - This will produce a log file for each query at `logs/<name>/[model]/[category]/[query_index].log` and a csv file `data/<name>.csv` (semicolon separated) with all the extracted data.
  - You can run the three methods in parallel using: `./run_pipeline.sh ae_tapaal ../bin/verifypn-tokelim-linux64 tapaal & ./run_pipeline.sh ae_dynamic ../bin/verifypn-tokelim-linux64 dynamic & ./run_pipeline.sh ae_static ../bin/verifypn-tokelim-linux64 static &`

This pipeline runs each query in `MCC2023-CTL` sequentially and will take a _very long time_.
Therefore, we have included the data files used in the paper in this reproducibility package: `data/demo_tapaal.csv`, `data/demo_dynamic.csv`, and `data/demo_static.csv`.
You may also consider producing the data partially. See the section below.

If you wish to rerun the entire benchmark, we recommend running the individual queries in parallel. However, we do not include scripts to do so.
See `scripts/run_single.sh` to run a single query and `scripts/extract.sh` to extract the data from the log files.
The time/memory limits can be found in `scripts/run_single.sh` (30 minutes and 15 GB by default).

### Reproduce data (partial)

*Expected run time: 3 hours per method*

Steps:

- Navigate to the `scripts/` directory: `cd scripts`
- Run pipeline: `./run_pipeline_partial.sh <name> ../bin/verifypn-tokelim-linux64 <method>` where `<name>` is the desired name of the output, and `<method>` is either `tapaal`, `dynamic`, or `static`. We advice to include the method in the name too, e.g. 'ae_tapaal'.
  - This will produce a log file for each query at `logs/<name>/[model]/[category]/[query_index].log` and a csv file `data/<name>.csv` (semicolon separated) with all the extracted data.
  - You can run the three methods in parallel using: `./run_pipeline_partial.sh ae_tapaal ../bin/verifypn-tokelim-linux64 tapaal & ./run_pipeline_partial.sh ae_dynamic ../bin/verifypn-tokelim-linux64 dynamic & ./run_pipeline_partial.sh ae_static ../bin/verifypn-tokelim-linux64 static &`

This partial pipeline runs the first query of the two CTL categories for every 50th model with a timeout of 10 minutes, a total of 56 queries.
Note that the shorter timeout may be a disadvantage for the dynamic and static token-elimination methods.

### Graphs and tables

*Expected run time: 10 seconds*

Steps:

- Navigate to the `scripts/` directory: `cd scripts`
- Run `python graphs_and_tables.py`
  - By default, the demo data (the data used in the paper) is used. To use your own data, the Python script must be
    given a series of arguments on the form 'name=file' where 'name' is the display name of the data and 'file' is
    the name of the file in `data/`. You must provide at least two data files and the first one must be named Tapaal
    as it will be used as the baseline for some graphs. Example:
    `python graphs_and_tables.py Tapaal=ae_tapaal.csv Other=ae_other.csv`.
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

### Scripts

The Bash and Python scripts in the directory `scripts/` are distributed under no license.
