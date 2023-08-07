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
:code:`timewise` is currently supported with python 3.10.

--------------------------
Using poetry (RECOMMENDED)
--------------------------
We recommend using poetry for python installation to properly resolve all dependencies (see `this <https://python-poetry.org>`_ for more info).

If you are using poetry to manage your python environment you can install timewise like this:

.. code-block:: console

   poetry add timewise

If you want to install :code:`timewise` in editable mode just clone the repository and execute this in the
:code:`timewise` root directory

.. code-block:: console

   poetry install

---------
Using pip
---------

.. code-block:: console

    pip install timewise

If you would like to install in editable mode just clone the repository and execute this in the :code:`timewise` root directory:

.. code-block:: console

   pip install -e

Note that the :code:`requirements.txt` was deprecated in :code:`v0.1.9`. The easiest way to install the dependencies is
to install via :code:`poetry`.

*************************************
Dependencies for showing SDSS cutouts
*************************************

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

*****
Usage
*****

How you want to access :code:`timewise` depends on your use case. If you want
to use the functionality off the shelf, head over to the CLI.
If you want to do more involved things or embed :code:`timewise` in your
code, check out the usage of the Classes directly.


****
Cite
****

If you use :code:`timewise` please cite `this <https://zenodo.org/badge/latestdoi/449677569>`_


********
CONTENTS
********

.. toctree::
   installation
   cli
   getting_started
   api
   :maxdepth: 1
