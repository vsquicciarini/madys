Installation
=====

.. _installation:

Install
------------

As catalog queries are mediated by the `TAP Gaia Query <https://github.com/mfouesneau/tap>`_ package (``TAP``), this module needs being present in your working environment. If you are trying to import ``MADYS`` from the command line, the module is automatically installed if not found. However, **this does not work on Jupyter Notebook**: in this case, we suggest to manually install the package from pip, through the following command:

.. code-block:: console

   pip install git+https://github.com/mfouesneau/tap
   
Please make sure **not** to write just `pip install tap`; this command would download **a different**, although eponymous, package. 

Also notice that :py:func:`TAP` might require the installation of `lxml <https://lxml.de/>`_.

.. note::

   If an error is thrown during the upload of ``lxml``, cancel the ``lxml`` package from your working path and force pip to install the v4.6.3, whose compatibility with TAP has been extensively tested.

After installing ``TAP``, the installation of ``MADYS`` can be smoothly performed through pip:

.. code-block:: console

   (.venv) $ pip install madys

Dependencies
----------------

Write dipendencies here

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

