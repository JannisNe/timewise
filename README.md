[![CI](https://github.com/JannisNe/timewise/actions/workflows/continous_integration.yml/badge.svg)](https://github.com/JannisNe/timewise/actions/workflows/continous_integration.yml)
[![Coverage Status](https://coveralls.io/repos/github/JannisNe/timewise/badge.svg?branch=main)](https://coveralls.io/github/JannisNe/timewise?branch=main)
[![PyPI version](https://badge.fury.io/py/timewise.svg)](https://badge.fury.io/py/timewise)
[![DOI](https://zenodo.org/badge/449677569.svg)](https://zenodo.org/badge/latestdoi/449677569)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JannisNe/timewise/examples?urlpath=tree/examples)



![](timewise.png)
# Infrared light curves from WISE data

This package downloads WISE data for positions on the sky and stacks single-exposure photometry per visit. It is designed to do so for efficiently for large samples of millions of objects.

## Prerequisites
Python version 3.11, 3.12 or 3.13.

If you want to not only download individual exposure photometry but also stack detections per visit (see below),
you must have access to a running [MongoDB](https://www.mongodb.com/)* **. 

<sub>* On MacOS have alook at the custom `brew` tap 
[here](https://github.com/mongodb/homebrew-brew)
to get the MongoDB community edition. </sub>

<sub>** On some systems this is not straight forward to set up. `timewise` requires it nevertheless as an integral part of the AMPEL system which is used to efficiently schedule and store the stacking of lightcurves. If you do not foresee a big overhead in calculating lightcurves for a sample of O(1000) objects, a more lightweight package might be more applicable. </sub>

## Installation

### If you use timewise only for downloading
The package can be installed via `pip` (but make sure to install the v1 pre-release):
```bash
pip install --pre timewise==1.0.0a10
```
### If you use timewise also for stacking individual exposures
You must install with the `ampel` extra:
```bash
pip install --pre 'timewise[ampel]==1.0.0a10'
```
To tell AMPEL which modules, aka units, to use, build the corresponding configuration file:
```bash
ampel config build -distributions ampel timewise -stop-on-errors 0 -out <path-to-ampel-config-file>
```

## Command line interface

Read the short description below or have a look at the example notebook(s): [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JannisNe/timewise/examples?urlpath=tree/examples)

```
 Usage: timewise [OPTIONS] COMMAND [ARGS]...                                                                
                                                                                                            
 Timewsie CLI                                                                                               
                                                                                                            
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --log-level           -l      TEXT  Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)                │
│                                     [default: INFO]                                                      │
│ --install-completion                Install completion for the current shell.                            │
│ --show-completion                   Show completion for the current shell, to copy it or customize the   │
│                                     installation.                                                        │
│ --help                              Show this message and exit.                                          │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────────────────────────────────╮
│ download        Download WISE photometry from IRSA                                                       │
│ prepare-ampel   Prepares the AMPEL job file so AMPEL can be run manually                                 │
│ process         Processes the lightcurves using AMPEL                                                    │
│ export          Write stacked lightcurves to disk                                                        │
│ run-chain       Run download, process and export                                                         │
│ plot            Make diagnostic plots                                                                    │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```

The input is a CSV file with at least three columns:  
- `orig_id`: an original identifier that **must** be an integer (for now)
- `ra`, `dec`: Right Ascension and Declination



`timewise` is configured with a YAML file. This is a sensible default which will use all single exposure photometry from AllWISE and NEOWISE:
```yaml
download:
  input_csv: <path-to-input>

  backend:
    type: filesystem
    base_path: <path-to-working-directory>

  queries:
    - type: positional
      radius_arcsec: 6
      table:
        name: allwise_p3as_mep
      columns:
        - ra
        - dec
        - mjd
        - cntr_mf
        - w1mpro_ep
        - w1sigmpro_ep
        - w2mpro_ep
        - w2sigmpro_ep
        - w1flux_ep
        - w1sigflux_ep
        - w2flux_ep
        - w2sigflux_ep

    - type: positional
      radius_arcsec: 6
      table:
        name: neowiser_p1bs_psd
      columns:
        - ra
        - dec
        - mjd
        - allwise_cntr
        - w1mpro
        - w1sigmpro
        - w2mpro
        - w2sigmpro
        - w1flux
        - w1sigflux
        - w2flux
        - w2sigflux

ampel:
  mongo_db_name: <mongodb-name>
```

This configuration file will be the input to all subcommands. Downloading and stacking can be run together or separate.


### To only download the data:
```bash
timewise download <path-to-config-file>
```
The photometry can be found in FITS files in the working directory specified in the configuration file\
along with metadata JSON files. These tell `timewise` which quries have already completed (per chunk) so the
download process can be interrupted and re-started at a later time.

### Stack individual exposure by visits
As mentioned above, this needs installation with the ampel extra.


To **execute the stacking** after the download:
```bash
timewise process <path-to-config-file> <path-to-ampel-config-file>
```

Make some **diagnostic plots** to check the datapoint selection and binning:
```bash
timewise plot <path-to-config-file> <indices-to-plot> <output-directory>
```

As a shortcut, you can also run **download, stacking, and export in one command**:
```bash
timewise run-chain <path-to-config-file> <path-to-ampel-config-file> <output-directory>
```

For more configuration options of the stacking, you can **run AMPEL manually**.

1. Prepare an AMPEL job file for stacking the single-exposure data:
```bash
timewise prepare-ampel <path-to-config-file>
```
The result will contain the path to the prepared AMPEL job file.

2. Run the AMPEL job
```bash
ampel job -config <path-to-ampel-config-file> -schema <path-to-ampel-job-file>
```





## Citation
If you use `timewise` please make sure to cite [Necker et al. A&A 695, A228 (2025)](https://www.aanda.org/articles/aa/abs/2025/03/aa51340-24/aa51340-24.html).
Additionally, you might want to include a reference to the specific version you are using: [![DOI](https://zenodo.org/badge/449677569.svg)](https://zenodo.org/badge/latestdoi/449677569)

## Difference lightcurves
Make sure to check out `timewise-sup`, the Timewise Subtraction Pipeline: 
[link](https://gitlab.desy.de/jannisnecker/timewise_sup).
