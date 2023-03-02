Handling photometry
=====

Quality cuts
------------

In ``MADYS``, particular attention is devoted to ensuring that only reliable photometric measurements are retained in the final database used for parameter determination. By default, ``SampleObject`` ’s mode 1 collects photometric measurements from Gaia DR2/DR3 (G, GBP, GRP) and 2MASS (J, H, Ks). Gaia DR3 G magnitudes are corrected by adopting the prescriptions by `Riello et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021A%26A...649A...3R/abstract>`_. As regards GBP and GRP, intrinsically much more sensitive than G to contamination from nearby sources or from the background, Gaia's phot_bp_rp_excess_factor C is used as a proxy to evaluate the quality of photometric measurements. In particular, following `Riello et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021A%26A...649A...3R/abstract>`_, we defined a color-independent corrected BP-RP excess factor C∗ for both Gaia DR2 and Gaia DR3 and excluded GBP and GRP magnitudes with a corrected excess factor larger, in absolute value, than 3 times the standard deviation of well-behaved sources of comparable magnitude (see `our paper <https://ui.adsabs.harvard.edu/abs/2022A%26A...666A..15S/abstract>`_ for details). 

For 2MASS and AllWISE, only sources with photometric flag ``ph_qual == 'A'`` are kept. If needed, a different value for the worst quality flag still considered reliable can be selected via the dedicated keywords ``max_tmass_q`` and ``max_wise_q``. From the documentation:


::

   * ``max_tmass_q`` (1): worst 2MASS photometric flag still considered reliable. Possible values, ordered by decreasing quality: 'A', 'B', 'C', 'D', 'E', 'F', 'U', 'X'. For a given choice, excludes all measurements with a lower quality flag. Default: 'A'.
   * ``max_wise_q`` (1): worst ALLWISE photometric flag still considered reliable. Possible values, ordered by decreasing quality: 'A', 'B', 'C', 'U', 'Z', 'X'. For a given choice, excludes all measurements with a lower quality flag. Default: 'A'.



To use Lumache, first install it using pip:

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
