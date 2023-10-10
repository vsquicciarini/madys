Instance creation
=====

``SampleObject`` is the class dedicated to catalog queries, computation of extinctions, and parameter estimation. In other words, a ``SampleObject`` instance is the container where our (sub)stellar sample of interest and its properties are defined and stored. Virtually any amount of objects (:math:`1 - \sim 10^5`) can be attached to an instance, allowing for a wide variety of scientific applications.

A ``SampleObject`` instance can be initialized in two different ways:

* mode 1: starting from object names, MADYS searchs for relevant information in all-sky catalogs;
* mode 2: the instance is completely determined by the data given as input.

In both cases, all the information related to the instance can be accessed through the instance attribute ``'phot_table'``.

From object names
------------

A valid input for mode 1 is, for instance, a .csv file. In this case, a column with IDs (named ``'source_id'``, ``'id'``, ``'ID'`` or ``'object_name'``) must be present. The type of IDs must be specified through the keyword ``id_type``. Possible values are ``'DR3'`` (if **all** the identifiers are taken from Gaia DR3), ``'DR2'`` (if **all** the identifiers are taken from Gaia DR2), or ``'other'`` (if **at least one** identifier refers to a different catalog, or if Gaia DR2 and Gaia DR3 identifiers are intermixed).

By default, ``MADYS`` cross-matches information from `Gaia DR2 <https://ui.adsabs.harvard.edu/#abs/2018A%26A...616A...1G/abstract>`_, `Gaia DR3 <https://ui.adsabs.harvard.edu/#abs/2022arXiv220800211G/abstract>`_ and `2MASS <https://ui.adsabs.harvard.edu/abs/2006AJ....131.1163S/abstract>`_. It is possible to specify a different set of catalogs through the keyword ``surveys``, provided that the additional entries belong to the following:

* ALLWISE (`Cutri et al. 2021 <https://ui.adsabs.harvard.edu/abs/2014yCat.2328....0C/abstract>`_);
* Panstarrs DR1 (`Chambers et al. 2016 <https://ui.adsabs.harvard.edu/abs/2016arXiv161205560C/abstract>`_);
* APASS DR9 (`Henden et al. 2016 <https://ui.adsabs.harvard.edu/abs/2016yCat.2336....0H/abstract>`_);
* SDSS DR13 (`Albareti et al. 2017 <https://ui.adsabs.harvard.edu/abs/2017ApJS..233...25A/abstract>`_).

.. note::

   If you wish to add a catalog to the query to use it for parameter determination, please ensure beforehand that its filters are available in the theoretical model of interest (cp. `Available filters <https://madys.readthedocs.io/en/latest/available_filters.html>`_).

The keyword ``ext_map`` selects the extinction map to be used to compute the integrated color excess E(B-V) for any object in the sample (cp. `Interstellar extinction <https://madys.readthedocs.io/en/latest/extinction_model.html>`_). Alternatively, a numpy array of E(B-V) with same size as the number of objects can be given via the keyword ``'ebv'``.

A minimal working example is given by:

.. code-block:: python

   file='1000stars.csv' #1000 random stars
   example_object=madys.SampleObject(file,id_type='DR3',ext_map='leike') 

From custom data
----------------

Mode 2 does not perform a query in all-sky catalogs, taking instead all the information it needs from the user. Data should be provided through an astropy table with column names corresponding to valid filters (see HERE). Photometric uncertainties corresponding to a filter ``'x'`` should be labeled as ``'x_err'``.

Using a real-life example, we provide here information about two planets in the direct-imaged HR8799 system (data from Zurlo et al. 2016):

.. code-block:: python

   J, dJ, H, dH, K, dK = 5.383, 0.027, 5.280, 0.018, 5.240, 0.018 #2MASS magnitude and uncertainties for the primary star
   par, par_err = 24.4620, 0.0455 #Gaia DR3 parallax

   # creates a Table with all photometric data
   dic = {'object_name': ['HR 8799 b','HR 8799 c'],
          'SPH_J': [14.39+J, 13.21+J],
          'SPH_H2': [12.80+H, 11.81+H],
          'SPH_H3': [12.50+H, 11.50+H],
          'SPH_K1': [11.91+K, 10.95+K],
          'SPH_K2': [11.73+K, 10.62+K],
          'SPH_J_err': [np.sqrt(dJ**2+0.09**2),np.sqrt(dJ**2+0.13**2)],
          'SPH_H2_err': [np.sqrt(dH**2+0.14**2),np.sqrt(dH**2+0.12**2)],
          'SPH_H3_err': [np.sqrt(dH**2+0.10**2),np.sqrt(dH**2+0.10**2)],
          'SPH_K1_err': [np.sqrt(dK**2+0.06**2),np.sqrt(dK**2+0.05**2)],
          'SPH_K2_err': [np.sqrt(dK**2+0.09**2),np.sqrt(dK**2+0.07**2)],
          'parallax': [par, par],
          'parallax_err': [par_err, par_err]
         }
   input_data=Table(dic)

   example_object=madys.SampleObject(input_data,ext_map='leike')

Customizable options
------------

A list of valid keywords that can be provided at inizialization can be found below. Here 1 stands for "only for mode 1", 2 for "only for mode 2".

* ``file`` (1): string or list, required. It can be either:

  - a string, giving the full path of the file containing target names;
  - a list of IDs. Gaia IDs must begin by ``'Gaia DR2'`` or ``'Gaia DR3'``.
* ``file`` (2): astropy Table, required. Table containing target names and photometric data.
* ``id_type`` (1): string, required. Type of IDs provided: must be one among ``'DR2'``, ``'DR3'`` or ``'other'``.
* ``ext_map``: string, required. Extinction map used. Possible values: ``'leike'``, ``'stilism'``, ``None`` (=no map is used).
* ``mock_file``: string, optional. Only used if ``file`` is a list or a table. Full path of a fictitious file, used to extract the working path and to name the outputs after it. If not set and ``verbose``:math:`\ge 1`, ``verbose`` is automatically changed to 0.
* ``surveys`` (1): list, optional. List of surveys whence photometric data is retrieved. Default: ``['gaia','2mass']``.
* ``get_phot`` (1): bool or string, optional. Set to:

  - True: to query the provided IDs;
  - False: to recover photometric data from a previous execution; the filename and path must match the default one (see documentation).
  - string: full path of the file to load photometric data from. The file should come from a previous execution.
  
   Default: True.
* ``simbad_query`` (1): bool, optional. Set to True to query objects without a 2MASS cross-match in SIMBAD. It can significantly slow down data queries. Default: True if n<100, False otherwise.
* ``ebv``: float or numpy array, optional. If set, uses the i-th element of the array as E(B-V) for the i-th star. Default: not set, computes E(B-V) through the map instead.
* ``max_tmass_q`` (1): worst 2MASS photometric flag still considered reliable. Possible values, ordered by decreasing quality: 'A', 'B', 'C', 'D', 'E', 'F', 'U', 'X'. For a given choice, excludes all measurements with a lower quality flag. Default: 'A'.
* ``max_wise_q`` (1): worst ALLWISE photometric flag still considered reliable. Possible values, ordered by decreasing quality: 'A', 'B', 'C', 'U', 'Z', 'X'. For a given choice, excludes all measurements with a lower quality flag. Default: 'A'.
* ``verbose``: int, optional. Degree of verbosity of the various tasks performed by MADYS. It can be:
  
  - 0: no file is saved, nothing is printed;
  - 1: a .csv file with retrieved information is saved (1), limited information is printed;
  - 2: in addition to the output of 1, a log file is created;
  - 3: in addition to the output of 2, .txt files are created when executing SampleObject.get_params().
  Default: 2. However, if file is a list or a table and mock_file is not set, it is forcedly set to 0.


Attributes
------------

Here we list the attributes of a ``SampleObject`` instance.

* ``file``: string. Corresponding to either ``file`` (1) or ``mock_file`` (2).
* ``path``: string. Working path, where all inputs and outputs can be found.
* ``log_file``: string. Name of the log_file (if ``verbose`` `\ge 2`). Open the file for details on the process.
* ``phot_table``: astropy Table. Contains all the information related to the instance.
* ``abs_phot``: numpy array. Absolute magnitudes in the filters of interest.
* ``abs_phot_err``: numpy array. Errors on absolute magnitudes in the filters of interest.
* ``par``: numpy array. Parallaxes of the objects.
* ``par_err``: numpy array. Errors on parallaxes.
* ``filters``: list. Set of filters, given either by the filters of ``surveys`` (1) or by column names (2).
* ``surveys`` (1): list. Surveys whence photometric data are retrieved.
* ``mode``: int. The execution mode.
* ``ID``: astropy Table. Original set of IDs.
* ``GaiaID``: astropy Table. Gaia IDs (original or recovered). If original, they can come from DR3 or DR2. If recovered, they always come from DR3.
* ``log_file``: Path object. Full path of the log file. Not set if verbose<2.

