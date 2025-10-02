[![CI](https://github.com/JannisNe/timewise/actions/workflows/continous_integration.yml/badge.svg)](https://github.com/JannisNe/timewise/actions/workflows/continous_integration.yml)
[![Coverage Status](https://coveralls.io/repos/github/JannisNe/timewise/badge.svg?branch=main)](https://coveralls.io/github/JannisNe/timewise?branch=main)
[![PyPI version](https://badge.fury.io/py/timewise.svg)](https://badge.fury.io/py/timewise)
[![DOI](https://zenodo.org/badge/449677569.svg)](https://zenodo.org/badge/latestdoi/449677569)


![](timewise.png)
# Infrared light curves from WISE data

This package downloads WISE data for positions on the sky and stacks single-exposure photometry per visit

## Prerequisites

`timewise` makes use of [AMPEL](https://ampelproject.github.io/ampelastro/) and needs a running [MongoDB](https://www.mongodb.com/).

## Installation
The package can be installed via `pip`:
```bash
pip install timewise
```

To tell AMPEL which modules, aka units, to use, build the corresponding configuration file:
```bash
ampel config build -distributions ampel timewise -stop-on-errors 0 -out <path-to-ampel-config-file>
```

## Usage

### Command line interface

```
 Usage: timewise [OPTIONS] COMMAND [ARGS]...                                    
                                                                                
 Timewsie CLI                                                                   
                                                                                
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --log-level           -l      TEXT  Logging level (DEBUG, INFO, WARNING,     │
│                                     ERROR, CRITICAL)                         │
│                                     [default: INFO]                          │
│ --help                              Show this message and exit.              │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ download        Download WISE photometry from IRSA                           │
│ prepare-ampel   Prepares the AMPEL job file so AMPEL can be run manually     │
│ process         Processes the lightcurves using AMPEL                        │
│ export          Write stacked lightcurves to disk                            │
│ run-chain       Run download, process and export                             │
╰──────────────────────────────────────────────────────────────────────────────╯

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


#### All-in-one:
Run download, stacking, and export:
```bash
timewise run-chain <path-to-config-file> <path-to-ampel-config-file> <output-directory>
```

#### Separate download and processing:
To only download the data:
```bash
timewise download <path-to-config-file>
```

To execute the stacking:
```bash
timewise process <path-to-config-file> <path-to-ampel-config-file>
```

#### Run AMPEL manually
Prepare an AMPEL job file for stacking the single-exposure data:
```bash
timewise prepare-ampel <path-to-config-file>
```
The result will contain the path to the prepared AMPEL job file that can be run with
```bash
ampel job -config <path-to-ampel-config-file> -schema <path-to-ampel-job-file>
```


## Citation
If you use `timewise` please make sure to cite [Necker et al. A&A 695, A228 (2025)](https://www.aanda.org/articles/aa/abs/2025/03/aa51340-24/aa51340-24.html).
Additionally, you might want to include a reference to the specific version you are using: [![DOI](https://zenodo.org/badge/449677569.svg)](https://zenodo.org/badge/latestdoi/449677569)

## Difference lightcurves
Make sure to check out `timewise-sup`, the Timewise Subtraction Pipeline: 
[link](https://gitlab.desy.de/jannisnecker/timewise_sup).
