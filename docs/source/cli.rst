CLI
===

If you do not require a custom implementation of a :code:`ParentSample` or :code:`WISEData`
but rather want to use the functionality off the shelf then you can
easily do so on the command line.

***********
Config File
***********

:code:`timewise` will read the instructions from a configuration :code:`YAML` file.
All possible contents are:

* :code:`base_name`: The name for your project, determines the output directory in the :code:`TIMEWISE_DATA` directory
* :code:`filename`: Path to a CSV file containing the parent sample
* :code:`class_name` (optional, default= :code:`WiseDataByVisit`): The name of the :code:`WISEData` class
* :code:`min_sep_arcsec`: The value for the separation [arcsec] to associate datapoints to a source
* :code:`n_chunks`: The number of chunks in which to split the sample when downloading. If you are looking at sample with <1e5 objects then one chunk is enough!
* :code:`default_keymap`: A mapping from :code:`ra`, :code:`dec` and :code:`id` to the respective columns in the CSV file, default to :code:`ra` = :code:`ra` etc.
* :code:`instructions`: A dictionary containing the methods of :code:`WISEData` you want to call and the respective arguments

An example is shown below:

.. code-block:: yaml

    base_name: test
    filename: <path/to/file>
    default_keymap:
     ra: RAdeg
     dec: DEdeg
     id: AllWISE_designation
    instructions:
     get_photometric_data:
      service: gator
     plot_lc:
      parent_sample_idx: 0
      service: gator
     plot_diagnostic_binning:
      ind: 0
      service: gator


********************
Command Line Command
********************

To execute the :code:`YAML` file, invoke the :code:`timewise` command:

.. code-block:: console

    timewise <path/to/yaml/file>
