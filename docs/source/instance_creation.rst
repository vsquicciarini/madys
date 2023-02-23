Instance creation
=====

``SampleObject`` is the class dedicated to catalog queries, computation of extinctions, and parameter estimation.

A ``SampleObject instance can be initialized in different ways (see below).

Two inizialization modes exist:

* mode 1: starting from object names, it searchs for relevant information in all-sky catalogs;
* mode 2: completely determined by the data given as input.

From object names
------------

A valid input for mode 1 is, for instance, a .csv file. A column with IDs (labeled as 'source_id', 'id', 'ID' or 'object_name') must be present.

The keyword ext_map selects the extinction map to be used to compute the integrated color excess E(B-V) for any object in the sample. Alternatively, a numpy array of E(B-V) with same size as the number of objects can be given via the keyword 'ebv'.

verbose=0 means that no output file is desired.

Output data are stored in an astropy table, accessible through the instance attribute 'phot_table'.

.. code-block:: console

   f='1000stars.csv' #1000 random stars
   p1=madys.SampleObject(f,id_type='DR3',ext_map='leike',verbose=0) #collects data 
   p1.phot_table

From custom data
----------------

To retrieve a list of random ingredients,
you can use the ``lumache.get_random_ingredients()`` function:

.. autofunction:: lumache.get_random_ingredients

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
will raise an exception.

.. autoexception:: lumache.InvalidKindError

For example:

>>> import lumache
>>> lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']

