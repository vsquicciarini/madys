Fitting unresolved binaries
=====

Starting from v1.3.0, MADYS provides a straightforward way to derive the parameters of both components of an unresolved binary system. Given the ubiquity of short-period binaries and the biases these systems introduce in any effort dealing with the age and mass distribution of a stellar sample, this feature of the program will be beneficial to a large variety of statistical studies focused on directly imaged exoplanets or on the stars themselves.

Given the limited information used my MADYS to derive stellar parameters, no attempt is made to leave the secondary-to-primary ratio as a free parameter; instead, the user can specify either a secondary-to-primary mass ratio (through the keyword ``secondary_q``) or a contrast ratio (through the keyword ``secondary_contrast``) in a suitable band when running the :py:func:`SampleObject.get_params` method.

Let's suppose that a ``SampleObject`` instance is creating containing a list of two stars. 

.. code-block:: python

   example_object = madys.SampleObject(file, id_type='DR3', verbose=0, ext_map='leike')

Using a scalar value for ``secondary_q`` will apply the given mass ratio to all the objects:

.. code-block:: python

   result = example_object.get_params('bhac15', secondary_q=0.7)

Using an array with same length as the object list will instead apply a different q to every system:

.. code-block:: python

   result = example_object.get_params('bhac15', secondary_q=np.array([0.7, 0.5]))

.. note::

   Unlike ``secondary_q``, ``secondary_contrast`` must be specified as a one-key dictionary with key name equal to the name of the filter where the contrast is measured. The dictionary value works exactly in the same way as a ``secondary_q`` value.
