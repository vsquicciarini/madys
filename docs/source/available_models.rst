Available models
=====

Nomenclature
------------

In its attempt to organize the jungle of evolutionary models published in the recent literature, MADYS establishes a taxonomical classification of models based on four levels of hierarchy: 

* the model family (L1), which can be thought as a container of all the models developed by a certain working team;
* the model suite (L2), that should be characterized by a well-defined set of physical assumptions, equations, atmospheric and evolutionary codes; in general, a suite of model is described by one or more publications;
* the model version (L3). The model suite can be distributed under different flavours, characterized by e.g. varying boundary conditions, treatments of dust, and "minor" differences;
* the model grid (L4), that adds to the previous one the specification of up to six input parameters (metallicity, helium fraction, rotational velocity, alpha enhancement, fraction of stellar surface covered by star spots, and magnetic field.

For example:

.. code-block::

  L1 = model_family: ATMO #(ERC-ATMO @ University of Exeter)
  L2 = model_suite: ATMO2020 #(suite published in Phillips et al. 2020)
  L3 = model_version: ATMO2020-ceq #(version assuming chemical equilibrium)
  L4 = model_grid: ATMO2020-ceq_p0.00 #(solar metallicity)

The class ``ModelHandler`` is meant to handle the model database of MADYS and to assist the user in:

* understanding which models are available;
* resolving the taxonomy of any of them;
* downloading the corresponding files;
* identifying the available filters and customizable parameters.

Many of these tasks are accomplished through the function :py:func:`ModelHandler.available`. In particular, the full model database with its taxonomic tree can be accessed through the command:

.. code-block:: python

   ModelHandler.available('full_model_list')

Calling the same function with argument equal to a valid model suite or version will return -- if at least one grid is locally available -- a detailed verbose description of the model itself, including literature references, customizable parameters, age and mass ranges, taxonomical information and the list of surveys currently featured in the model itself. The range of age, mass and parameters is dynamically adjusted based on the files present in the working path of MADYS, giving precise information on what is actually available for use in MADYS. 

.. code-block:: python

   ModelHandler.available('mist')

  
  # isochrone model: MIST (v1.2)
  # MESA revision number = 7503
  # Basic references: Dotter, ApJS, 222, 8 (2016) + Choi et al., ApJ 823, 102 (2016)
  # Solar mixture: Y = 0.2703, Z = 0.0142
  # Photometric systems: 
  # 2mass, bessell, gaia, hipparcos, hr, kepler, sdss, tess, tycho, wise
  # Mass range (M_sun): [0.1, 149.5237]
  # Age range (Myr): [0.1, 19952.6]
  # Available metallicities: [-4.0,-3.5,-3.0,-2.5,-2.0,-1.75,-1.5,-1.25,-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5]
  # Available rotational velocities: [0.0,0.4]
  # Available alpha enhancements: [0.0]
  # Available magnetic field strengths: [0]
  # Available spot fractions: [0.0]
  # Available helium contents: [0.2703]
  # Model family: mist
  # Model suite: mist
  # Call it as: 'mist'

If the function is called with no argument, all verbose information about all locally available model suites will be printed at once.


Usage
------------
Whenever a model is required as input of a MADYS function, the user should intend it as a model version. The parameters uniquely specifying the model grid to be internally employed by the program must be supplied, if needed, by means of optional keywords named ``feh``, ``afe``, ``he``, ``v_vcrit``, ``fspot``, and ``B``.

Model grids constitute the input of just one function, :py:func:`ModelHandler.download_model`; additionally, they can be accepted as input of :py:func:`ModelHandler.available`.


Download models
------------
A user can in any moment download a model grid through the following function:

.. code-block:: python

   ModelHandler.download_model(model_grid)

This function is automatically called when attempting to use a combination of parameters that is best reproduced by a model grid that is not available in the current working path of MADYS. In this case, the program will ask whether to use the local best-matching model or to download and use the more suitable model available in the Zenodo repository associated to MADYS.


Model extrapolation
------------
Starting from v2.0.0, all models in MADYS can be linearly extrapolated outside their dynamical range. This functionality was introduced to meet a need of the JWST community, which often stumbles across situations where the performances of JWST are so deep that they breach the mass/temperature range where substellar evolutionary models are computed.

Given a DetectionMap `instance <https://madys.readthedocs.io/en/latest/contrast_curves.html>`_, extrapolation can be turned by means of the ``extrapolate`` keyword:

.. code-block:: python

   dpm = instance.DImode_from_contrasts('atmo2023-ceq', plot=True, extrapolate=True)

The option is inherited from the ``IsochroneGrid``, and it can be used there as well.

.. note::

   As usual with extrapolation, this feature must be used with a grain of salt. The reason why models are not computed outside certain temperature ranges is because some of their assumptions — related to physical or chemical processes, or to the neglect of certain species — break down. In addition to this, the linear extrapolation of magnitudes is a rough first order approximation, sacrificing accuracy for numerical robustness.


Piecewise models
------------
For the same reasons underlying the extrapolation feature, we provide a custom model that is assembled in a piecewise way by connecting the BEX suite ($M ∈ [0.02, 2]~M_J$) and the ATMO 2023 suite ($M ∈ [0.5, 75]~M_J$). An example of an interpolated mass track is shown here:

.. image:: images/bex-atmo_interpolation.png

For the BEX part, we employ the bex-petitcode-clear version; for ATMO, we leave to the user the choice among the three versions available therein.

Two additional models named BEX-Dusty and BEX-Cond, employed in the framework of the `SHINE <https://ui.adsabs.harvard.edu/abs/2021A%26A...651A..72V/abstract>`_ survey, are provided too.



