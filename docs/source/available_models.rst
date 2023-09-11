Available models
=====

Nomenclature
------------

In its attempt to organize the jungle of evolutionary models published in the recent literature, MADYS establishes a taxonomical classification of models based on four levels of hierarchy. 

The first level is the **family** of models ( ∼
  the same team behind). The second corresponds to a suite of models (those found in papers;  ∼
  fixed input physics). The third is the version of the model, which might come in different flavours with varying boundary conditions. The fourth is the specific grid, defined by values of astrophysical parameters (e.g. metallicity). The last one corresponds to the file name.

So, I have:

L1 = model_family = "family" L2 = model_suite = "suite" L3 = model_version = "version" L4 = model_grid = "grid"

Example:

model_family: ATMO
model_suite: ATMO2020
model_version: ATMO2020-ceq
model_grid: ATMO2020-ceq_p0.00
For each of these levels, I want to define useful functions:

model_family: which names, versions and grids belong to the family? --> available()
model_suite: what is the age range, mass range and the list of filters for models in the suite? What are the customizable parameters? What are the available versions?
model_version: same as above, apart from the last one ("available grids?"). Same functions should work with both;
model_grid: what is the full path to the file? How can I extract data?
Then, I need some functions to go from one level to another. Family does not need such a function, because it is never called explicitly (apart from the "available" function, which just lists its sublevels). Beware that the folders containing the models are named after families.

Suites are never called by the user.

model_version -> model_grid is performed by version_to_grid (old name: _grid_from_parameters), which takes as additional input a dictionary of parameters which uniquely determines the grid.

model_grid -> model_version is performed by grid_to_version (old name: _parameters_from_grid), which returs both model_version and the dictionary of parameters which uniquely determines the grid.

The two functions above are one the inverse of the other.

Versions shall be used as input in all the functions called by the user. Grids can be called only within available and within a function that downloads that model.


.. code-block:: console

   (.venv) $ pip install lumache

Creating recipes
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
