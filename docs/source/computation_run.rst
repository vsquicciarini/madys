Parameter determination
=====

Usage
------------

The core of MADYS is represented by the function :py:func:`SampleObject.get_params`, which performs parameter determination upon an existing ``SampleObject`` instance. A minimal working code snippet is given by the following example:


.. code-block:: python

   example_object = madys.SampleObject(file, id_type='DR3', verbose=0, ext_map='leike')
   result = example_object.get_params('bhac15')
   
that will compute the relevant parameters using model bhac15 and store them in an instance of the FitParams class (see below).

.. note::

   The list of currently available models can be retrieved through the function ModelHandler.available(). Please have also a look to `this page <https://madys.readthedocs.io/en/latest/available_models.html>`_ to have a detailed overview of the naming convention adopted in MADYS, which follows a taxonomic convention with four levels of hierarchy.

Several theoretical parameters can be specified at this point through dedicated keywords, allowing one to choose the available grid that best suites the sample under examination:

* metallicity (``feh``);
* helium fraction (``he``);
* rotational velocity (``v_vcrit``);
* alpha enhancement (``afe``);
* fraction of stellar surface covered by star spots (``fspot``);
* magnetic field (``B``).

If, e.g., there is a list of stars with different metallicities, a vector of ``feh`` parameters can be provided so as to compare every star with a model of adequate metallicity. Not every model allows for the possibility to tune each of this parameters; additionally, MADYS does not attempt to interpolate along these parameter axis, but rather looks for the closest matching model in its database. If, for instance, the following command is executed: 

.. code-block:: python

   result = example_object.get_params('parsec', feh = 0.3, fspot = 0.2)

MADYS will identify as closest model the PARSEC v1.2 grid with [Fe/H]=+0.25; the star spot parameter will be ignored because it's not currently supported by PARSEC isochrones. The parameter values actually used in the computation will be accessible as homonymous attributes of the ``FitParams`` instance "result".


Custom settings:


The FitParams class
----------------



