Usage
=====


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

**************
Data directory
**************

`timewise` is capable of downloading a lot of data. You can specify the location of all that data:

.. code-block:: console

    export TIMEWISE_DATA=/path/to/data/directory

******************
Available Services
******************

The data access is courtesy of IRSA. The recommended service for small parent samples (<300 sources) is
`gator <https://irsa.ipac.caltech.edu/applications/Gator/GatorAid/irsa/catsearch.html>`_. For larger querie use
`tap <https://irsa.ipac.caltech.edu/docs/program_interface/TAP.html>`_