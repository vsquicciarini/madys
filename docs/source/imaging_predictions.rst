Imaging Survey Predictions
=====

Let's consider again a basic example such as the ones described `here <https://madys.readthedocs.io/en/latest/instance_creation.html>`_:

.. code-block:: python

   example_object = madys.SampleObject(file, id_type='DR3', verbose=0, ext_map='leike')
   result = example_object.get_params('mist')
   
Let us suppose that the objects of our interest come equipped with Gaia and 2MASS photometry, and that we are interested in estimating WISE W1 and W2 magnitudes. This can be done with the keyword ``additional_columns``:

.. code-block:: python

   example_object = madys.SampleObject(file, id_type='DR3', verbose=0, ext_map='leike')
   result = example_object.get_params('mist', additional_columns = ['W1', 'W2'])

``MADYS`` will compute the required magnitudes, starting from the best-fit solution for the observed photometry of the star. The outputs will be stored as attributes of the resulting ``FitParams`` instance; a "synth_" prefix is added to the name of the quantity to clarify its synthetic nature. In other words, in the case above the W1 photometry will be stored as ``results.synth_W1``, its 16th/84th percentiles being ``results.synth_W1_min``/``results.synth_W1_max``, respectively.

This feature is particularly interesting in the context of direct imaging studies. Let us suppose a giant planet, discovered in the K magnitude. Is it worth asking for observational time to detect the planet in, e.g., J or H magnitudes? By providing an easy way to provide photometric predictions, ``MADYS`` can help users to understand whether such an object would lie above or below the detection limits of their instrument in the band of interest.
