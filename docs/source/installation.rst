Get Set-Up
==========

**************
Data directory
**************

:code:`timewise` is capable of downloading and handling a lot of data.
You can specify the location of the working directory:

.. code-block:: console

    export TIMEWISE_DATA=/path/to/data/directory

Here you will also find all plots etc.

If you are using the class :code:`WISEDataDESYCluster` you have the possibility to specify another directory where the final
products will be moved to:

.. code-block:: console

    export TIMEWISE_BIGDATA=/path/to/bigdata/directory

******************
Available Services
******************

The data access is courtesy of IRSA. The recommended service for small parent samples (<300 sources) is
`gator <https://irsa.ipac.caltech.edu/applications/Gator/GatorAid/irsa/catsearch.html>`_. For larger queries use
`tap <https://irsa.ipac.caltech.edu/docs/program_interface/TAP.html>`_