# Inferring Concept Drift Without Labeled Data

This repo accompanies the Cloudera Fast Forward Labs research report - [FF22: Inferring Concept Drift Without Labeled Data](https://concept-drift.fastforwardlabs.com/). The content of the repository supports the Experiments section of our report and includes: all data preprocessing steps used to induce concept drift on the [Covertype Dataset](https://archive.ics.uci.edu/ml/datasets/covertype), implementations of the three methods discussed in the report, a test harness for evaluating each method in an adaptive learning workflow with programmable parameter settings, and interactive notebooks for running/evaluating drift detection methods in the test harness.

## Repo Structure

```
.
├── data                                          # contains raw and post-processed data
├── logs                                          # holds logfiles after experiments are run
├── notebooks                                     # contains all notebooks that demonstrate basic usage
├── requirements.txt                              # python dependencies
├── test_harness                                  # main library of code supporting experiments
├── LICENSE
└── README.md
```

### `notebooks`

```
├── Covertype_EDA.ipynb                           # details data preprocessing and drift induction methods
├── Covtype_experiment_dev.ipynb                  # demonstration of how to interact with the test_harness library
└── archive                                       # directory full of working/development notebooks
```

### `test_harness`

```
├── datasets
│   ├── __init__.py
│   └── dataset.py														# class with methods for advancing windows through eximents
├── experiments
│   ├── __init__.py
│   ├── base_experiment.py                    # abstract base class for all experiments to inherit
│   ├── baseline_experiment.py                # baseline case from report
|   ├── topline_experiment.py                 # topline case from report
|   ├── response_uncertainty_experiment.py    # method 2 from report
|   ├── margin_uncertainty_experiment.py      # method 3 from report
│   └── margin_threshold_experiment.py        # method 4 from report
└── utils
    └── utils.py														  # plotting and reporting utilities 
```

This directory provides the main library for interacting with drift detection methods, experiments, and plotting utlities. Refer to the report for details on each experimental method.

## Project Setup

To reproduce the experiments install the project requirements and then run the notebooks toggling parameters as desired. The code here was developed against Python 3.8.8, and likely also to functions with more recent versions of Python.

To install dependencies, first create and activate a new virtual environment through your preferred means, then pip install from the requirements file:

```
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

In CML or CDSW, no virtual env is necessary. Instead, inside a Python 3 session (with at least 2 vCPU / 4 GiB Memory), simply run

```
!pip3 install -r requirements.txt     # notice `pip3`, not `pip`
```

