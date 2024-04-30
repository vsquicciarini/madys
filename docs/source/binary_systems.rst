Parameter determination
=====

Usage
------------

The core of MADYS is represented by the function :py:func:`SampleObject.get_params`, which performs parameter determination upon an existing ``SampleObject`` instance. A minimal working code snippet is given by the following example:


.. code-block:: python

   example_object = madys.SampleObject(file, id_type='DR3', verbose=0, ext_map='leike')
   result = example_object.get_params('bhac15')
   
that will compute the relevant parameters using model bhac15 and store them in an instance of the ``FitParams`` class (see below).

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


Additional parameters can be optionally specified. They are meant to allow users to gauge the age range, the mass range, the computational speed, the management of photometric uncertainties and the outputs according to their needs:

* mass_range: list, optional. A two-element list with minimum and maximum mass within the grid (M_sun). Default: not set; the mass_range is the intersection between a rough mass estimate based on G and K magnitudes and the dynamical range of the model itself.
* age_range: list or numpy array, optional. It can be either:

    * a two-element list with minimum and maximum age to consider for the whole sample (Myr);
    * a 1D numpy array, so that the i-th age (Myr) is used as fixed age for the i-th star;
    * a 2D numpy array with 2 columns. The i-th row defines (lower_age,upper_age) range in which one or more solutions are found for the i-th star.
    * a 2D numpy array with 3 columns. The i-th row is used as (mean_age,lower_age,upper_age) for the i-th star; mean_age is used as in case 2), and [lower_age, upper_age] are used as in case 3).
  Default: [1,1000].

* n_steps: list, optional. Number of (mass, age) steps of the interpolated grid. Default: [1000,500].
* n_try: int, optional. Number of Monte Carlo iteractions for each star used to take photometric uncertainties into account. Default: 1000.
* fill_value: array-like or (array-like, array_like) or “extrapolate”, optional. How the interpolation over mass deals with values outside the original range. Default: np.nan. See scipy.interpolate.interp1d for details.
* ph_cut: float, optional. Maximum  allowed photometric uncertainty on absolute magnitudes [mag]. Photometric measurements with a larger error will be ignored. Default: 0.2.
* m_unit: string, optional. Unit of measurement of the resulting mass. Choose either 'm_sun' or 'm_jup'. Default: 'm_sun'.
* save_maps: bool, optional. Set to True to save chi2 and weight maps for each star. Not recommended if n_star is big (let's say, >1000). Default: False.
* logger: logger, optional. A logger returned by SampleObject._setup_custom_logger(). Default: self.__logger.


The FitParams class
----------------
The output of :py:func:`SampleObject.get_params` is an instance of the ``FitParams`` class. Let us explore the attributes of this class:


* ``ages``: numpy array. Final age estimates [Myr].
* ``ages_min``: numpy array. Minimum age (given by the user or derived) [Myr].
* ``ages_max``: numpy array. Maximum age (given by the user or derived) [Myr].
* ``masses``: numpy array. Final mass estimates [M_sun or M_jup].
* ``masses_min``: numpy array. Minimum mass estimates [M_sun or M_jup].
* ``masses_max``: numpy array. Maximum mass estimates [M_sun or M_jup].
* ``ebv``: numpy array. Adopted/computed E(B-V), one element per star [mag].
* ``ebv_err``: numpy array. Error on E(B-V), null if not explicitly set at initialization.
* ``chi2_min``: numpy array. Reduced chi2 of best-fit solutions.
* ``radii``: numpy array. Final radius estimates [R_sun or R_jup].
* ``radii_min``: numpy array. Minimum radius estimates [R_sun or R_jup].
* ``radii_max``: numpy array. Maximum radius estimates [R_sun or R_jup].
* ``logg``: numpy array. Final surface gravity estimates [log10([cm s-2])].
* ``logg_min``: numpy array. Minimum surface gravity estimates [log10([cm s-2])].
* ``logg_max``: numpy array. Maximum surface gravity estimates [log10([cm s-2])].
* ``logL``: numpy array. Final luminosity estimates [log10([L_sun])].
* ``logL_min``: numpy array. Minimum luminosity estimates [log10([L_sun])].
* ``logL_max``: numpy array. Maximum luminosity estimates [log10([L_sun])].
* ``Teff``: numpy array. Final effective temperature estimates [K].
* ``Teff_min``: numpy array. Minimum effective temperature estimates [K].
* ``Teff_max``: numpy array. Maximum effective temperature estimates [K].
* ``fit_status``: numpy array. Flag for the outcome of the fitting process, one element per star.

   * 0: successful fit.
   * 1: all magnitudes for the star have an error beyond the maximum allowed threshold: age and mass determinations was not possible.
   * 2: all magnitudes for the star are more than 0.2 mag away from their best theoretical match. Check age and mass range of the theoretical grid, or change the model if the current one does not cover the expected age/mass range for this star.
   * 3: no point with chi2<1000 was found for the star.
   * 4: the third closest filter in the best-fit solution is more than 3 sigma away from its theoretical match, and the third closest magnitude to its theoretical match is more than 0.1 mag away.
   * 5: undefined error.
* ``chi2_maps``: list. Only present if save_maps=True in the parent analysis. Contains one 2D numpy array per star; matrix elements are reduced chi2 estimates for grid points, using nominal data.
* ``weight_maps``: list. Only present if save_maps=True in the parent analysis. Contains one 2D numpy array per star; matrix elements are the weight of grid points, as used to obtain the final family of solutions.
* ``all_solutions``: list. Contains a dictionary per star, with all possible solutions providing an accettable fit to data.
* ``feh``: float. [Fe/H] of the grid.
* ``he``: float. Helium content of the grid.
* ``afe``: float. Alpha enhancement [a/Fe] of the grid.
* ``v_vcrit``: float. Rotational velocity of the grid.
* ``fspot``: float. Fraction of stellar surface covered by star spots.
* ``B``: int. Whether magnetic fields are included (1) or not (0) in the grid.
* ``sample_name``: string. Only returned if verbose>0. Name of the sample file, without extension.
* ``path``: string. Only returned if verbose>0. Full path to the sample file, without extension.
* ``objects``: numpy array. List of analyzed objects. Corresponds to self.Gaia_ID of the parent ``SampleObject`` instance.
* ``exec_command``: list. Each entry is the __repr__ of the IsochroneGrid object used within :py:func:`SampleObject.get_params`.
* ``fitting_mode``: int. Fitting mode of the parent :py:func:`SampleObject.get_params` process. It can be either:

   * 1: the age was set a priori to a single value, or the selected model_version only has one age; corresponding to case 2) for the keyword 'age_range' from SampleObject.get_params.
   * 2: the age was to be found within the specified interval; corresponding to case 1) or 3) for the keyword 'age_range' from SampleObject.get_params.
   * 3: the age was fixed, and age_min and age_max were used to compute errors; corresponding to case 4) for the keyword 'age_range' from SampleObject.get_params.
   * 4: the program was left completely free to explore the entire age range.
* ``model_grid``: list. Each entry is the model_version used to fit the corresponding star with :py:func:`SampleObject.get_params`.
* ``is_true_fit``: bool. Whether the instance comes directly from a fit, or if it's an average of different model estimates.


Averaging parameter estimates
----------------
Starting from v1.0.1, a function :py:func:`SampleObject.average_results` allows the direct average of two or more ``FitParams`` instances coming from the same underlying ``SampleObject`` instance. It is meant to give the user a way to estimate the inter-model dispersion by computing the mean and standard deviation of a sample of parameter estimates for the same input objects. For instance, the following set of commands:

.. code-block:: python

   star_obj = SampleObject(star_list,ext_map='leike',id_type='other')
   result1 = star_obj.get_params('parsec',age_range=[1,100])
   result2 = star_obj.get_params('mist',age_range=[1,100],feh=np.array([-0.9,0]))
   averaged_res = FitParams.average_results([result1,result2])

will analyze the sample contained in star_obj twice and then create a new ``FitParams`` instance containing averaged results.

.. note::

   Unlike the general approach of MADYS, this function naively assumes that every parameter of every best-fit solution can be approximated by a normal distribution and that parameter uncertainties across different instances are equivalent (i.e., an arithmetic mean is performed). These two approximations might not always hold, especially if the age is not well constrained. Hence, it is strongly adviced to use this function with caution.
