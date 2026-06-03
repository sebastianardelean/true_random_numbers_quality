# Introduction to Quantum Computing (Notebook)


## Description

This repository accompanies the textbook "Introduction to Quantum Computing". It contains all the implementations in Python of the topics discussed in the notebook.

## Installation

Clone or download this repository and change directory to `environment`.

### Using Anaconda/Miniconda or Venv

1. Install miniconda
2. Create the environment `conda env create -f environment.yml`
3. Activate the newly created environment
4. Install the necessary libraries: `pip install -r requirements.txt`

### Using Docker

1. Install Docker
2. Run `docker build -t quantum .`
3. Run `docker run -p 8888:8888 -it -v <path-to-workspace>:/app --entrypoint /bin/bash quantum`