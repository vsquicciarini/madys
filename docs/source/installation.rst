Installation
=====

.. _installation:

Install
------------

The installation of ``MADYS`` can be performed through pip:

.. code-block:: console

   (.venv) $ pip install madys

Dependencies
----------------

Write dipendencies here and xlml problem

To retrieve a list of random ingredients,
you can use the :py:func:`lumache.get_random_ingredients` ``lumache.get_random_ingredients()`` function:

.. autofunction:: lumache.get_random_ingredients

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
will raise an exception.

.. autoexception:: lumache.InvalidKindError

For example:

>>> import lumache
>>> lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']

