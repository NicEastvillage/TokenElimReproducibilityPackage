# Token Elimination Reproducibility Package

This is a reproducibility package for the paper 'Token Elimination in Model Checking of Petri Nets' by Nicolaj Ø. Jensen, Jiri Srba, and Kim G. Larsen.

The package contains the models, queries, binaries, and data used to produce the results of the paper as well as scripts to reproduce the results.

## Prerequisites

- Bash
- Python 3.10
- tar

## Instructions

### Setup
- Setup Python virtual environment: `python -m venv .venv`
- Activate the virtual environment: `source .venv/bin/activate`
- Install Python packages: `pip install -r requirements.txt`
- Extract models and queries: `tar -xvf MCC2023-CTL.tar.gz`

### Reproduce data (optional)
- Navigate to the `scripts/` directory: `cd scripts`
- Run pipeline: `./run_pipeline.sh <name> ../bin/verifypn-tokelim-linux64 <method>` where `<name>` is the desired name of the output, and `<method>` is either `tapaal`, `dynamic`, or `static`.
  - This will produce a log file for each query in the `logs/` directory and a csv file `data/<name>.csv` (semicolon separated) with all the extracted data in `data/`.

This pipeline runs each query in `MCC2023-CTL` sequentially and will take a _very long time_.
Therefore, we have included the data files used in the paper in this reproducibility package: `data/demo_tapaal.csv`, `data/demo_dynamic.csv`, and `data/demo_static.csv`.

If you wish to rerun the entire benchmark, we recommend running the queries in parallel. See `scripts/run_single.sh` to run a single query and `scripts/extract.sh` to extract the data from the log files.
The time/memory limits can be found in `scripts/run_single.sh` (30 minutes and 15 GB by default).

### Graphs and tables
- Navigate to the `scripts/` directory: `cd scripts`
- Run `python graphs_and_tables.py`
  - This will produce the graphs and tables from the paper in `output/`.
