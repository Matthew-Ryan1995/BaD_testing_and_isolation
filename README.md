# BaD_testing_and_isolation
This repository contains the python code to reproduce the results from *A Behaviour and Disease Model of Testing and Isolation*.

**Please note the following**

- All code was developed using Python 3.10.12
- All code was developed using Mac-standard file paths
- The python code assumes that `./python` is the working directory and uses relative file paths from this location.s

**Repository description**

1. `python` - this directory contains all python code associated to the article.
  - `BaD.py` contains the model class and all convenience functions.
  - `0*_*.py` files contain the scripts to reproduce the figures from the article.
  - `model_parameters.json` contains the default model parameters for the model to laod in.
2. `img` contains all the outputted figures from the python scripts.  To save figures here, ensure the flag `save_plot_flag` is set to `True`, otherwise figures will output in the python session.
3. `outputs` contains all saved simulation outputs from the python scripts.  To ensure that simulation results are produced and saved out, make sure the `generate_data_flag` is set to `True` in each python script.
- `requirements.txt` contains the python dependencies used in the environment where this code was developed.
