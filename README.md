[![CI](https://github.com/JannisNe/timewise/actions/workflows/continous_integration.yml/badge.svg)](https://github.com/JannisNe/timewise/actions/workflows/continous_integration.yml)
[![Coverage Status](https://coveralls.io/repos/github/JannisNe/timewise/badge.svg?branch=main)](https://coveralls.io/github/JannisNe/timewise?branch=main)
[![PyPI version](https://badge.fury.io/py/timewise.svg)](https://badge.fury.io/py/timewise)
[![Documentation Status](https://readthedocs.org/projects/timewise/badge/?version=latest)](https://timewise.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/449677569.svg)](https://zenodo.org/badge/latestdoi/449677569)



# `timewise` is great, love it!
Download infrared lightcurves recorded with the WISE satellite. More info soon!

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

### Data directory
``timewise`` is capable of downloading a lot of data. You can specify the location of all that data:
```
export TIMEWISE_DATA=/path/to/data/directory
```

### The `ParentSample` class 

To tell `timewise` which data you want to download, you have to create a subclass of `ParentSampleBase`. 
The subclass has to define two key attributes:
* `ParentSample.df`: A `pandas.DataFrame` consisting of minimum three columns: two columns holding the sky positions of each object in the form of right ascension and declination and one row with a unique identifier.
* `ParentSample.default_keymap`: a dictionary, mapping the column in `ParentSample.df` to 'ra', 'dec' and 'id'

Further, `ParentSampleBase` requires a `base_name` determining the location of any data in the `timewise` data directory.

```python
from timewise import ParentSampleBase
import pandas as pd


class MyParentSample(ParentSampleBase):

    default_keymap = {
        'ra': 'RA',
        'dec': 'DEC',
        'id': 'Name'
    }

    def __init__(self):
        self.df = pd.DataFrame(
            {'RA': [1, 2, 3], 'DEC':[-5, 0, 5], 'Name':['Wolf359', 'Vulcan', 'Kamino']}       
        )
        base_name = 'weird_sources'
        super().__init__(base_name=base_name)
```


### The ``WISEData`` class

This is the class that implements all core functionality:
* match your catalogue to WISE sources
* download photometric data
* bin the photometric data

Any `WISEData` class must be derived from `timewise.WISEDataBase` and implement the methods `bin_lightcurves()` and 
``_calculate_metadata()``

When initialising an instance of the class you need following arguments:
* ``parent_sample_class``: your class of parent sample (Attention: yes tha class and not an instance!)
* ``min_sep_arcsec``: the separation from your parent sample source where you want to look for WISE data
* ``n_chunks``: number of chunks into which your parent sample data will be split
* ``base_name``: same as for the parent sample

Currently there are two usable classes:

* ``timewise.WiseDataByVisit``: bins the photometric data by the "visit" of WISE to each sky position. 
These are periods when the sky position is observed by WISE and consists typically 
of few tens of observations each six months. 
The metadata that is calculated gives some basic measures on the variability.

* ``timewise.WISEDataDESYCluster``: derived from ``timewise.WiseDataByVisit``, uses the DESY cluster in Zeuthen 
to do the binning

Continuing from the example above let's use that parent sample to download the corresponding data:

```python
from timewise import ParentSampleBase, WiseDataByVisit
import pandas as pd


base_name = 'weird_sources'


class MyParentSample(ParentSampleBase):

    default_keymap = {
        'ra': 'RA',
        'dec': 'DEC',
        'id': 'Name'
    }

    def __init__(self):
        self.df = pd.DataFrame(
            {'RA': [1, 2, 3], 'DEC':[-5, 0, 5], 'Name':['Wolf359', 'Vulcan', 'Kamino']}       
        )
        super().__init__(base_name=base_name)

wd = WiseDataByVisit(
    base_name=base_name,
    min_sep_arcsec=8,
    parent_sample_class=MyParentSample,
    n_chunks=1
)

# matches the parent sample to sources in the AllWISE source catalog
wd.match_all_chunks(table_name="AllWISE Source Catalog")

# load photometric data 
wd.get_photometric_data(
    tables=None,            # query the default tables 'AllWISE Multiepoch Photometry Table' and 'NEOWISE-R Single Exposure (L1b) Source Table'
    perc=1,                 # get 100% of the data
    wait=0,                 # wait 0 hours bewteen queries
    service=None,           # use the dafault service, options are 'gator' (recommended for <300 sourecs) and 'tap'
    chunks=None,            # default is to download all chunks
    overwrite=False,        # overwrite any data that was previously downloaded
)

# plot some results
wd.plot_lc(
    parent_sample_idx=0,        # the index in the parent sample
    service='gator',            # use data downloaded with this service
    plot_unbinned=False,        # plot unbinned data as well
    plot_binned=True,           # plot the binned data
    interactive=False,          # if True, assumes you're in a Jupyter Notebook and return the Figure and axes
    fn='0_flux_desnity.pdf',    # filename for saving
    ax=None,                    # any pre-existing axes you want to plot in
    save=True,                  # if True saves the figure
    lum_key='flux_density'      # can also be 'mag'
                                # and **kwargs will be passed to plt.subplots()
)
```