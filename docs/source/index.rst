.. timewise documentation master file, created by
   sphinx-quickstart on Mon Jan 24 15:47:54 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to timewise's documentation!
====================================

.. image:: https://github.com/JannisNe/timewise/actions/workflows/continous_integration.yml/badge.svg
    :target: https://github.com/JannisNe/timewise/actions/workflows/continous_integration.yml

.. image:: https://coveralls.io/repos/github/JannisNe/timewise/badge.svg?branch=main
    :target: https://coveralls.io/github/JannisNe/timewise?branch=main

.. image:: https://badge.fury.io/py/timewise.svg
    :target: https://badge.fury.io/py/timewise

.. image:: https://zenodo.org/badge/449677569.svg
   :target: https://zenodo.org/badge/latestdoi/449677569





************
Installation
************

.. code-block:: console

    pip install timewise

If you would like to contribute just clone the repository. Easy.

************
Dependencies
************

All dependencies are listed in `requirements.txt`. If installing with `pip` they will automatically installed.
Otherwise you can install them with `pip install -r requirements.txt`.

There is one package that does not obey! It's `SciServer`!
It's used to access SDSS data and plot cutouts. If you want to use this functionality
install `this <https://github.com/sciserver/SciScript-Python>`_ and create an account `here <https://www.sciserver.org)>`_.
As soon as required you will be required to enter your username and password.

*******
Testing
*******

You can verify that everything is working (because this package is flawless and works everywhere.) by executing
the unittest

.. code-block:: console

    python -m unittest discover tests/


****
Cite
****

If you use `timewise` please cite `this <https://zenodo.org/badge/latestdoi/449677569>`_


********
CONTENTS
********

.. toctree::
   installation
   getting_started
   api
   :maxdepth: 1
