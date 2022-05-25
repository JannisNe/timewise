[![CI](https://github.com/JannisNe/timewise/actions/workflows/continous_integration.yml/badge.svg)](https://github.com/JannisNe/timewise/actions/workflows/continous_integration.yml)
[![Coverage Status](https://coveralls.io/repos/github/JannisNe/timewise/badge.svg?branch=main)](https://coveralls.io/github/JannisNe/timewise?branch=main)
[![PyPI version](https://badge.fury.io/py/timewise.svg)](https://badge.fury.io/py/timewise)
[![Documentation Status](https://readthedocs.org/projects/timewise/badge/?version=latest)](https://timewise.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/449677569.svg)](https://zenodo.org/badge/latestdoi/449677569)



# `timewise` is great, love it!
Download infrared lightcurves recorded with the WISE satellite.

## Installation

`timewise` is a python package, installable through `pip`
```
pip install timewise
```

If you would like to contribute just clone the repository. Easy.


## Dependencies

All dependencies are listed in `requirements.txt`. If installing with `pip` they will automatically installed.
Otherwise you can install them with `pip install -r requirements.txt`.

There is one package that does not obey! It's `SciServer`! 
It's used to access SDSS data and plot cutouts. If you want to use this functionality 
install [this](https://github.com/sciserver/SciScript-Python) and create an account [here](https://www.sciserver.org).
As soon as required you will be required to enter your username and password.


## Testing
 You can verify that everything is working (because this package is flawless and works everywhere.) by executing
 the unittest
```
python -m unittest discover tests/
```

## Cite
If you you `timewise` please cite [this](https://zenodo.org/badge/latestdoi/449677569).

## Usage
Detailed documentation can be found [here](https://timewise.readthedocs.io/en/latest/)
