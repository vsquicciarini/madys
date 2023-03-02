Parameter determination
=====

Usage
------------

The core of MADYS is represented by the function :py:func:`SampleObject.get_params`, which performs parameter determination upon an existing ``SampleObject`` instance. A minimal working code snippet is given by the following example:


.. code-block:: python

   example_object=madys.SampleObject(file,id_type='DR3',verbose=0,ext_map='leike')
   result=example_object.get_params('bhac15')
   
that will compute the relevant parameters using model bhac15 and store them in an instance of the FitParams class (see HERE).

Custom settings:


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
