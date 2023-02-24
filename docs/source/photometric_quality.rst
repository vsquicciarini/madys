Handling photometry
=====

Quality cuts
------------

In ``MADYS`` we give particular attention Particular attention 


This Section describes the conditions for a photometric measurement to be retained in the final database. By default, madys’
mode 1 collects photometric measurements from Gaia DR2/DR3
(G, GBP, GRP) and 2MASS (J, H, Ks). Gaia DR3 G magnitudes are corrected by adopting the prescriptions by Riello et al.
(2021). As regards GBP and GRP, which are known to be intrinsically much more sensitive than G to contamination from
nearby sources or from the background (Evans et al. 2018), the
phot_bp_rp_excess_factor C is used as a proxy to evaluate the
quality of photometric measurements. Following Riello et al.
(2021), a color-independent corrected BP=RP excess factor C∗
was defined for both Gaia DR2 and Gaia DR3:
C∗ = C + k0 + k1∆G + k2∆G2 + k3∆G3 + k4G (1)
where ∆G = (GBP − GRP).
The corrected BP=RP excess factor has an expected value
of 0 for well-behaved sources at all magnitudes but, when considering subsamples of stars with similar brightness, it tends to
widen out for fainter G; a varying standard deviation σ(G) can
be defined (Riello et al. 2021) as follows:
σC∗(G) = c0 + c1 · Gm: (2)
Values for the constants for Eq. 1- 2 are taken from Riello et al.
(2021) for DR3 and Squicciarini et al. (2021) for DR2, and are
provided in Table 1.
We exclude GBP and GRP magnitudes with a corrected excess
factor larger, in absolute value, than 3 σC∗(G). As mentioned
above, a value of C∗ significantly different from zero might be
due to blended Gaia transits or crowding effects; in addition to
this, it can also be related to an over-correction of the background (if C∗<0) or to an anomalous SED (if C∗>0) characterized by strong emission lines in the wavelength window where
the GRP transmissivity is larger than the G transmittivity. This
latter case can occur, for instance, for a source located in a HII
region (see discussion in Riello et al. 2021).
From 2MASS and AllWISE, only sources with photometric
flag ph_qual=’A’ are kept. If needed, a different value for the
worst quality flag still considered reliable can be selected




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
